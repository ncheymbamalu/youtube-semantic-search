import numpy as np
import polars as pl

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# read in ./app/data/transcript_embeddings.parquet as a pl.LazyFrame
df: pl.LazyFrame = pl.scan_parquet("./app/data/transcript_embeddings.parquet")

# load the embedding model
model: SentenceTransformer = SentenceTransformer("./app/artifacts/embedding_model")

def get_indices(query: str, k: int = 5) -> list[int]:
    """
    Returns the indices of the top k transcripts that have the strongest
    contextual relationship (cosine similarity score) with the input query

    Args:
        query (str): Input query
        k (int): Number of indices to return. Defaults to 5.

    Returns:
        list[int]: Top k indices
    """
    # compute the cosine similarity between the input query's embeddings and ...
    # the embeddings of each transcript
    cols: list[str] = [col for col in df.columns if col.startswith("embedding")]
    scores: np.ndarray = cosine_similarity(
        model.encode(query).reshape(1, -1), df.select(cols).collect().to_numpy()
    )

    # extract the index of the top k transcripts
    indices: list[int] = np.argsort(scores).ravel().tolist()[::-1][:k]
    return indices
