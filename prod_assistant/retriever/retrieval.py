# ── Imports ───────────────────────────────────────────────────────────────
import os                                              # read environment variables (API keys, DB tokens)
from langchain_astradb import AstraDBVectorStore       # the vector database client (our AstraDB store)
from utils.config_loader import load_config            # helper that reads config.yaml into a dict
from utils.model_loader import ModelLoader             # helper that loads the embedding model + LLM
from dotenv import load_dotenv                          # loads secrets from the .env file into the environment
from langchain.retrievers.document_compressors import LLMChainFilter        # uses an LLM to drop irrelevant docs
from langchain.retrievers import ContextualCompressionRetriever            # wraps a retriever + a compressor
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy  # RAGAS metrics
# Add the project root to the Python path for direct script execution
# project_root = Path(__file__).resolve().parents[2]   # (disabled) would compute the project root
# sys.path.insert(0, str(project_root))                 # (disabled) would add it to Python's search path


# ── The Retriever class: fetches relevant reviews from the vector DB ──────────
class Retriever:
    def __init__(self):
        """_summary_
        """
        self.model_loader = ModelLoader()      # prepares the embedding model + LLM loaders
        self.config = load_config()            # read config.yaml (collection name, top_k, etc.)
        self._load_env_variables()             # load + validate the secret keys from .env
        self.vstore = None                     # placeholder — the vector store (built later, once)
        self.retriever_instance = None         # placeholder — the retriever object (built later, once)

    def _load_env_variables(self):
        """_summary_
        """
        load_dotenv()                          # read the .env file into the environment

        # The secrets this retriever cannot run without.
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]

        # Collect any required variable that is missing (not set).
        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        # If even one is missing, stop immediately with a clear error.
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        # Save each secret onto the object so other methods can use them.
        self.google_api_key = os.getenv("GOOGLE_API_KEY")                    # Google key (for embeddings)
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")            # AstraDB URL
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")  # AstraDB auth token
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")                    # AstraDB keyspace/namespace

    def load_retriever(self):
        """_summary_
        """
        # Build the vector store only once (skip if already built).
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]  # which collection to search

            # Connect to AstraDB — the same store the ingestion step wrote data into.
            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),  # turns the query into a vector
                collection_name=collection_name,                # collection name
                api_endpoint=self.db_api_endpoint,              # DB URL
                token=self.db_application_token,                # auth token
                namespace=self.db_keyspace,                     # keyspace/namespace
                )
        # Build the retriever only once (skip if already built).
        if not self.retriever_instance:
            # How many results to return: read top_k from config, else default to 3.
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3

            # MMR = Maximal Marginal Relevance: fetches results that are both relevant AND diverse
            # (avoids returning near-duplicate reviews).
            mmr_retriever = self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k,          # final number of docs to return
                                "fetch_k": 20,       # fetch 20 candidates first, then pick the best k
                                "lambda_mult": 0.7,  # 0=max diversity, 1=max relevance (0.7 leans relevant)
                                "score_threshold": 0.6  # ignore matches weaker than this score
                               })
            print("Retriever loaded successfully.")   # small confirmation message

            llm = self.model_loader.load_llm()         # load the LLM (used by the compressor below)

            # LLMChainFilter uses the LLM to READ each retrieved doc and DROP the ones that
            # aren't actually relevant to the query — an extra quality filter after MMR.
            compressor = LLMChainFilter.from_llm(llm)

            # Combine the two: MMR fetches candidates, then the compressor filters them.
            self.retriever_instance = ContextualCompressionRetriever(
                base_compressor=compressor,     # the LLM-based relevance filter
                base_retriever=mmr_retriever    # the MMR retriever that feeds it candidates
            )

        return self.retriever_instance          # hand back the ready-to-use retriever

    def call_retriever(self, query):
        """_summary_
        """
        retriever = self.load_retriever()        # get a ready retriever (builds it on first call)
        output = retriever.invoke(query)          # run the search: fetch + filter docs for the query
        return output                             # return the list of matching Document objects


# ── Run this block only when the file is executed directly (not imported) ─────
if __name__ == '__main__':
    user_query = "Is the Lenovo IdeaPad Slim 5 a good laptop?"  # the test question

    retriever_obj = Retriever()                          # create the retriever (loads config/secrets/models)

    retrieved_docs = retriever_obj.call_retriever(user_query)  # fetch the most relevant review docs

    # Show the question and the documents that were retrieved for it.
    print(f"\n=== Query ===\n{user_query}\n")
    print(f"=== Retrieved {len(retrieved_docs)} document(s) ===")
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"\n--- Result {idx} ---")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content)

    # Helper: turn the retrieved Document objects into readable text blocks.
    def _format_docs(docs) -> str:
        if not docs:                                     # nothing found
            return "No relevant documents found."
        formatted_chunks = []
        for d in docs:                                   # for each retrieved doc...
            meta = d.metadata or {}                      # its product info (title, price, rating)
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"   # product title (or N/A if missing)
                f"Price: {meta.get('price', 'N/A')}\n"           # price
                f"Rating: {meta.get('rating', 'N/A')}\n"         # rating
                f"Reviews:\n{d.page_content.strip()}"            # the review text itself
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)       # join all docs with a separator

    retrieved_contexts = [_format_docs([doc]) for doc in retrieved_docs]  # format each doc into text (pass a 1-item list)

    #this is not an actual output this have been written to test the pipeline
    response="The Lenovo IdeaPad Slim 5 is a good budget laptop with solid performance and battery life under 1,00,000 INR." # fake answer for testing

    # Score the pipeline with RAGAS: how precise was the retrieval, and how relevant is the answer.
    context_score = evaluate_context_precision(user_query,response,retrieved_contexts)
    relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_contexts)

    print("\n--- Evaluation Metrics ---")
    print("Context Precision Score:", context_score)     # higher = retrieved docs were more relevant
    print("Response Relevancy Score:", relevancy_score)  # higher = answer was more relevant to the question





    # for idx, doc in enumerate(results, 1):
    #     print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")
