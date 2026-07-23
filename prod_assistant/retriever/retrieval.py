# ── Imports ─────────────────────────────────────────────────────────────────
import os                                              # read environment variables (API keys, DB tokens)
import sys                                             # lets us modify Python's module search path (sys.path)
from langchain_astradb import AstraDBVectorStore       # the vector database client (our AstraDB store)
from typing import List                                # used only for type hints (e.g. List[Document])
from langchain_core.documents import Document          # the "Document" object each review is stored as
from utils.config_loader import load_config            # helper that reads config.yaml into a dict
from utils.model_loader import ModelLoader             # helper that loads the embedding model
from dotenv import load_dotenv                         # loads secrets from the .env file into the environment
import sys                                             # (duplicate import of sys — harmless, already imported above)
from pathlib import Path                               # for building file paths in an OS-independent way

# Add the project root to the Python path for direct script execution.
# __file__ = this file's path; .resolve() makes it absolute;
# parents[2] climbs 2 folders up: retriever -> prod_assistant -> project root.
project_root = Path(__file__).resolve().parents[2]     # = /Users/aryan/ecomm-prod-assistant
# Insert the project root at the FRONT of the search path so imports can be found
# even when you run this file directly (python path/to/retrieval.py).
sys.path.insert(0, str(project_root))

# ── The Retriever class: fetches relevant reviews from the vector DB ──────────
class Retriever:
    def __init__(self):
        """_summary_
        """
        # Prepare the embedding-model loader (turns text into vectors).
        self.model_loader = ModelLoader()
        # Read config.yaml (collection name, top_k, etc.) into a dictionary.
        self.config = load_config()
        # Load + validate the secret keys/tokens from the .env file.
        self._load_env_variables()
        # Placeholders — filled in later by load_retriever(). None = "not built yet".
        self.vstore = None                             # will hold the AstraDB vector store
        self.retriever = None                          # will hold the search object

    def _load_env_variables(self):
        """_summary_
        """
        # Read the .env file and load its values into os.environ.
        load_dotenv()

        # The secrets this program cannot run without.
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]

        # Build a list of any required variable that is missing (not set).
        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        # If even one is missing, stop immediately with a clear error.
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        # Save each secret onto the object so other methods can use them.
        self.google_api_key = os.getenv("GOOGLE_API_KEY")            # Google API key (for embeddings)
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")    # AstraDB URL
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")  # AstraDB auth token
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")            # AstraDB namespace/keyspace

    def load_retriever(self):
        """_summary_
        """
        # Build the vector store only once (if not already built).
        if not self.vstore:
            # Get the collection (table) name from config.yaml -> astra_db.collection_name.
            collection_name = self.config["astra_db"]["collection_name"]

            # Connect to AstraDB. This is the same store the ingestion step wrote data into.
            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),  # model to turn the query into a vector
                collection_name=collection_name,                # which collection to search
                api_endpoint=self.db_api_endpoint,              # where the DB lives
                token=self.db_application_token,                # auth token
                namespace=self.db_keyspace,                     # keyspace/namespace
            )
        # Build the retriever object only once (if not already built).
        if not self.retriever:
            # How many results to return: read top_k from config, else default to 3.
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3

            # Turn the vector store into a "retriever" that returns the top_k matches.
            retriever = self.vstore.as_retriever(
                search_kwargs={"k": top_k})                     # k = number of results to fetch
            print("Retriever loaded successfully.")             # small confirmation message
            return retriever                                    # hand the retriever back to the caller

    def call_retriever(self, query):
        """_summary_
        """
        # Get a ready-to-use retriever (builds it on first call).
        retriever = self.load_retriever()
        # Run the actual search: find the reviews most relevant to `query`.
        output = retriever.invoke(query)
        # Return the list of matching Document objects.
        return output


# ── Run this block only when the file is executed directly (not imported) ─────
if __name__ == '__main__':
    # Create the retriever object (loads config, secrets, model loader).
    retriever_obj = Retriever()
    # The question we want to search reviews for.
    user_query = "Can you suggest good budget iPhone under 1,00,00 INR?"
    # Run the search and get back the most relevant reviews.
    results = retriever_obj.call_retriever(user_query)

    # Loop over each result (idx starts at 1 for nicer numbering) and print it.
    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")




    # for idx, doc in enumerate(results, 1):
    #     print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")
