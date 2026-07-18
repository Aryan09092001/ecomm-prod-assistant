import csv
import time
import re
import os
import sys
import platform
import subprocess
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# --- Apple Silicon compatibility patch for undetected_chromedriver ---
# uc 3.5.5 (latest) always downloads the Intel (mac-x64) chromedriver on macOS,
# even on arm64 Macs, causing: OSError [Errno 86] Bad CPU type in executable.
# This wraps its platform detection so Apple Silicon gets the native driver.
# No-op on Intel Macs, Linux and CI, so it is safe to keep in the repo.
_uc_orig_set_platform_name = uc.patcher.Patcher._set_platform_name


def _uc_patched_set_platform_name(self):
    _uc_orig_set_platform_name(self)
    if sys.platform.endswith("darwin") and platform.machine() == "arm64":
        self.platform_name = "mac_arm64" if self.is_old_chromedriver else "mac-arm64"


uc.patcher.Patcher._set_platform_name = _uc_patched_set_platform_name

# uc rewrites bytes inside the chromedriver binary, which invalidates its macOS
# code signature; the OS then kills the driver with SIGKILL (exit code -9).
# Re-apply an ad-hoc signature after each patch so the binary can run again.
# macOS-only; a no-op (best effort) on other platforms.
_uc_orig_patch_exe = uc.patcher.Patcher.patch_exe


def _uc_patched_patch_exe(self):
    result = _uc_orig_patch_exe(self)
    if sys.platform.endswith("darwin"):
        try:
            subprocess.run(
                ["codesign", "--force", "--sign", "-", self.executable_path],
                check=True,
                capture_output=True,
            )
        except Exception:
            pass
    return result


uc.patcher.Patcher.patch_exe = _uc_patched_patch_exe


def _detect_chrome_major():
    """Return the installed Chrome major version so uc downloads a matching
    driver. Prevents "session not created: ChromeDriver only supports Chrome
    version X" when the latest driver is ahead of the installed browser.
    Returns None (uc falls back to latest) if detection fails.
    """
    candidates = []
    if sys.platform.endswith("darwin"):
        candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    elif sys.platform.startswith("win"):
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
    else:
        candidates = ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]
    for path in candidates:
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True, check=True).stdout
            match = re.search(r"(\d+)\.\d+\.\d+", out)
            if match:
                return int(match.group(1))
        except Exception:
            continue
    return None


_CHROME_MAJOR = _detect_chrome_major()
# --- end patch ---

class FlipkartScraper:
    # Flipkart auto-generates (and frequently renames) its CSS class names, and
    # uses different ones per product category / A-B test. Each field lists the
    # known selectors newest-first; _first_text tries them in order.
    TITLE_SELECTORS = ["div.KzDlHZ", "a.wjcEIp", "div.RG5Slk", "a.WKTcLC", "div.syl9yP"]
    PRICE_SELECTORS = ["div.Nx9bqj", "div.hZ3P6w", "div._30jeq3"]
    RATING_SELECTORS = ["div.XQDdHH", "div.MKiFS6", "span.CjyrHS", "div._3LWZlK"]
    REVIEW_SELECTORS = ["span.Wphh3N", "span.PvbNMB"]
    # Review cards on the product page (also frequently renamed).
    REVIEW_BLOCK_SELECTORS = [
        "div._1psv1ze9r._1o6mltljo",              # current (2026) atomic classes
        "div._27M-vq", "div.col.EPCmJX", "div._6K-7Co",  # older layouts
    ]

    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _first_text(element, selectors, default=""):
        """Return the text of the first matching selector, else `default`."""
        for selector in selectors:
            found = element.find_elements(By.CSS_SELECTOR, selector)
            if found and found[0].text.strip():
                return found[0].text.strip()
        return default

    @classmethod
    def _extract_reviews(cls, soup, count):
        """Pull up to `count` review texts from a product-page soup.

        Tries the known review-card selectors first; if Flipkart has renamed
        them again, falls back to anchoring on the "Verified Buyer" marker that
        every review carries.
        """
        blocks = []
        for selector in cls.REVIEW_BLOCK_SELECTORS:
            blocks = soup.select(selector)
            if blocks:
                break

        if not blocks:
            for node in soup.find_all(string=lambda s: s and "Verified Buyer" in s):
                parent = node.parent
                for _ in range(8):
                    if parent is None:
                        break
                    text = parent.get_text(" ", strip=True)
                    if 40 < len(text) < 600:
                        blocks.append(parent)
                        break
                    parent = parent.parent

        seen, reviews = set(), []
        for block in blocks:
            text = block.get_text(separator=" ", strip=True)
            # skip empties, dups, and the ratings-summary ("based on N ratings...")
            if not text or text in seen or text.lower().startswith("based on"):
                continue
            reviews.append(text)
            seen.add(text)
            if len(reviews) >= count:
                break
        return reviews

    def get_top_reviews(self,product_url,count=2):
        """Get the top reviews for a product.
        """
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = uc.Chrome(options=options,use_subprocess=True,version_main=_CHROME_MAJOR)

        if not product_url.startswith("http"):
            driver.quit()
            return "No reviews found"

        try:
            driver.get(product_url)
            time.sleep(4)
            try:
                driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
                time.sleep(1)
            except Exception:
                pass  # login/offer popup not always present; ignore silently

            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            reviews = self._extract_reviews(soup, count)
        except Exception:
            reviews = []

        driver.quit()
        return " || ".join(reviews) if reviews else "No reviews found"
    
    def scrape_flipkart_products(self, query, max_products=1, review_count=2):
        """Scrape Flipkart products based on a search query.
        """
        options = uc.ChromeOptions()
        driver = uc.Chrome(options=options,use_subprocess=True,version_main=_CHROME_MAJOR)
        search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(4)

        try:
            driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
        except Exception:
            pass  # login/offer popup not always present; ignore silently

        time.sleep(2)
        products = []

        items = driver.find_elements(By.CSS_SELECTOR, "div[data-id]")
        for item in items:
            if len(products) >= max_products:
                break
            try:
                title = self._first_text(item, self.TITLE_SELECTORS)
                price = self._first_text(item, self.PRICE_SELECTORS, "N/A")
                rating = self._first_text(item, self.RATING_SELECTORS, "N/A")
                reviews_text = self._first_text(item, self.REVIEW_SELECTORS)
                match = re.search(r"\d+(,\d+)?(?=\s+Reviews)", reviews_text)
                total_reviews = match.group(0) if match else "N/A"

                link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                href = link_el.get_attribute("href")
                # Skip cards that are not actual products (ads, banners, etc.)
                if not title or not href:
                    continue
                product_link = href if href.startswith("http") else "https://www.flipkart.com" + href
                match = re.findall(r"/p/(itm[0-9A-Za-z]+)", href)
                product_id = match[0] if match else "N/A"
            except Exception as e:
                print(f"Error occurred while processing item: {e}")
                continue

            top_reviews = self.get_top_reviews(product_link, count=review_count) if "flipkart.com" in product_link else "Invalid product URL"
            products.append([product_id, title, rating, total_reviews, price, top_reviews])

        driver.quit()
        return products
    
    def save_to_csv(self, data, filename="product_reviews.csv"):
        """Save the scraped product reviews to a CSV file."""
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):  # filename includes subfolder like 'data/product_reviews.csv'
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # plain filename like 'output.csv'
            path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)
        