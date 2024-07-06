import numpy as np
import polars as pl

from sentence_transformers import SentenceTransformer

from src.config import Config
from src.logger import logging


def encode_transcripts(
        input_file: str,
        output_file: str,
        model_name: str = "thenlper/gte-base",
        model_dir: str = "embedding_model"
) -> None:
    """Converts YouTube video transcripts (text) to embeddings (vectors)

    Args:
        input_file (str): Name of the file that contains the video transcripts
        output_file (str): Name of the file that the embeddings will be written to
        model_name (str): Name of the embedding model. Defaults to 'thenlper/gte-base'.
        model_dir (str): Name of the directory the embedding model will be saved to.
        Defaults to 'embedding_model'.
    """
    try:
        # instantiate the embedding model and extract its embedding dimension
        model: SentenceTransformer = SentenceTransformer(model_name)
        dmodel: int = model.get_sentence_embedding_dimension()

        # read in the data containing the YouTube video transcripts as a pl.DataFrame
        df: pl.DataFrame = pl.read_parquet(Config.Path.DATA_DIR / input_file)

        # convert each transcript to a dmodel-length vector of embeddings
        # NOTE: the 'embeddings' np.ndarray has shape (N, dmodel), that is, ...
        # N embedding vectors, one per transcript, where each has length dmodel
        embeddings: np.ndarray = model.encode(df["transcript"].to_list())
        schema: dict[str, type] = dict(zip(
            [f"embedding_{idx + 1}" for idx in range(dmodel)], [float] * dmodel
        ))

        # save the 'model' object to ../artifacts/
        logging.info(
            "Saving the embedding model, '%s', to %s",
            model_name, Config.Path.ARTIFACTS_DIR / model_dir
        )
        model.save(str(Config.Path.ARTIFACTS_DIR / model_dir))

        # horizontally concatenate the 'data' pl.DataFrame with a pl.DataFrame that ...
        # contains the embeddings, and then write to ../data/ as a parquet
        (
            pl.concat((df, pl.DataFrame(embeddings, schema=schema)), how="horizontal")
            .write_parquet(Config.Path.DATA_DIR / output_file)
        )
        logging.info(
            "Transcript embeddings have been generated and written to %s.",
            Config.Path.DATA_DIR / output_file
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    encode_transcripts("video_transcripts.parquet", "transcript_embeddings.parquet")
