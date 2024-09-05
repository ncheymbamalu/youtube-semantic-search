import os
import sys

import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.etl import latest_data
from src.paths import PathConfig, load_config

load_dotenv(PathConfig.PROJECT_DIR / ".env")


# set the 'TOKENIZERS_PARALLELISM' environment variable to false
# NOTE: this disables parallelism after forking the 'huggingface/tokenizers' repo
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM")

# fetch the embedding model from Hugging Face
embedding_model: SentenceTransformer = SentenceTransformer(
    load_config().get("embedding_model_id"), trust_remote_code=True
)


def get_indices(query: str, k: int = 5) -> list[int]:
    """Returns the index of the top k transcripts that have the strongest contextual
    relationship, i.e., highest cosine similarity score, with the input query

    Args:
        query (str): Input query
        k (int, optional): Number of indices to return. Defaults to 5.

    Returns:
        list[int]: Top k indices
    """
    try:
        # get the query's embeddings
        # NOTE: the resulting np.ndarray has shape (1, dmodel), ...
        # where dmodel is the embedding model's dimensionality
        query_embeddings: np.ndarray = embedding_model.encode(query).reshape(1, -1)

        # get the (N, dmodel) np.ndarray of transcript embeddings, ...
        # where N is the number of transcripts
        transcript_embeddings: np.ndarray = np.vstack(
            [
                np.frombuffer(serialized_embedding, dtype=np.float32)
                for serialized_embedding in latest_data["embedding"]
            ]
        )

        # compute the cosine similarity between the input query's embeddings and ...
        # the embeddings of each transcript
        scores: np.ndarray = cosine_similarity(query_embeddings, transcript_embeddings).ravel()

        # extract the index of the top k transcripts
        indices: list[int] = np.flip(scores.argsort()).tolist()[:k]
        return indices
    except Exception as e:
        raise e


def answer_query(query: str) -> None:
    """Prints out the title and url of the five YouTube videos whose transcript
    has the strongest contextual relationship with the input query

    Args:
        query (str): Input query
    """
    idx: list[int] = get_indices(query)
    titles: list[str] = latest_data[idx, "title"].to_list()
    urls: list[str] = [f"https://youtu.be/{video_id}" for video_id in latest_data[idx, "video_id"]]
    results: str = "\n".join(
        f"{i+1}. {title}: {url}" for i, (title, url) in enumerate(zip(titles, urls))
    )
    print(f"Query: {query}\n\nYouTube Results:\n{results}")


if __name__ == "__main__":
    # NOTE: useful resource, https://www.youtube.com/watch?v=1tFB2ux9DWU&t=1s
    task: str = sys.argv[1]
    user_query: str = sys.argv[2]
    answer_query(user_query) if task == "answer query" else print("Invalid task, please try again.")
