# ── Imports ───────────────────────────────────────────────────────────────
import asyncio                                          # to run RAGAS's async scoring from normal (sync) code
from utils.model_loader import ModelLoader              # our helper that loads the LLM + embedding model
from ragas import SingleTurnSample                      # RAGAS object that bundles one Q + answer + contexts
from ragas.llms import LangchainLLMWrapper              # wraps a LangChain LLM so RAGAS can use it
from ragas.embeddings import LangchainEmbeddingsWrapper # wraps a LangChain embedding model for RAGAS
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy  # the two metrics we compute
import grpc.experimental.aio as grpc_aio               # Google's async gRPC (needed by the Google client)

grpc_aio.init_grpc_aio()      # initialize async gRPC once, so the Google API client works without errors
model_loader = ModelLoader()  # create one shared model loader for both functions below


# ── Metric 1: Context Precision ─────────────────────────────────────────────
# "Were the retrieved documents actually relevant/useful for answering the query?"
def evaluate_context_precision(query, response, retrieved_context):
    try:
        # Bundle the question, the answer, and the retrieved docs into one RAGAS sample.
        sample = SingleTurnSample(
            user_input=query,                  # the user's question
            response=response,                 # the answer that was produced
            retrieved_contexts=retrieved_context,  # the docs the retriever returned
        )

        # RAGAS scoring is asynchronous, so we wrap it in an async function.
        async def main():
            llm = model_loader.load_llm()                        # load the LLM (used as the "judge")
            evaluator_llm = LangchainLLMWrapper(llm)             # wrap it for RAGAS
            context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)  # the metric
            result = await context_precision.single_turn_ascore(sample)  # await = run the async score
            return result                                        # a number (higher = better precision)

        # asyncio.run(...) starts an event loop, runs main() to completion, and returns its result.
        return asyncio.run(main())
    except Exception as e:
        # On any error, return the exception object instead of crashing (basic safety net).
        return e


# ── Metric 2: Response Relevancy ────────────────────────────────────────────
# "Is the generated answer actually relevant to the user's question?"
def evaluate_response_relevancy(query, response, retrieved_context):
    try:
        # Same bundling of question + answer + retrieved docs.
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )

        # Again, RAGAS scores asynchronously, so wrap the work in an async function.
        async def main():
            llm = model_loader.load_llm()                          # LLM used to judge relevancy
            evaluator_llm = LangchainLLMWrapper(llm)               # wrap the LLM for RAGAS
            embedding_model = model_loader.load_embeddings()       # this metric ALSO needs embeddings
            evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)  # wrap the embeddings
            # strictness=1 → generate only 1 question (Groq rejects n>1, unlike OpenAI)
            scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1)  # the metric
            result = await scorer.single_turn_ascore(sample)       # run the async score and wait for it
            return result                                          # a number (higher = more relevant)

        return asyncio.run(main())   # start the event loop, run main(), return the score
    except Exception as e:
        return e                     # return the error instead of crashing
