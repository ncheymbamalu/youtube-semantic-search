import polars as pl

from dotenv import load_dotenv
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.logger import logging
from src.paths import PathConfig, load_config
from src.utils import embed_transcripts, transcribe_videos

load_dotenv(PathConfig.PROJECT_DIR / ".env")

# load the latest data
latest_data: pl.DataFrame = pl.read_parquet(PathConfig.PROCESSED_DATA_PATH)


def etl(youtube_channel_ids: list[str], embedding_model_id: str) -> None:
    """Extracts video transcripts from a list of YouTube channel IDs, generates embeddings
    for the transcripts, and writes the resulting data to ~/data/youtube_transcripts.parquet

    Args:
        youtube_channel_ids (list[str]): YouTube channel IDs
        model_id (str): Name of the text embedding model.
    """
    try:
        # a list of records (YouTube video IDs) that have already been processed
        processed_video_ids: list[str] = latest_data["video_id"].unique().to_list()

        # iterate over each YouTube channel ID and create a pl.LazyFrame that contains
        # the video ID, creation date, title, and transcript of its 50 most recent videos
        lazyframes: list[pl.LazyFrame] = Parallel(n_jobs=-1)(
            delayed(transcribe_videos)(youtube_channel_id)
            for youtube_channel_id in tqdm(youtube_channel_ids)
        )

        # vertically concatenate the pl.LazyFrames in the 'lazyframes' list, and ...
        # only keep the records that have not been processed
        df: pl.DataFrame = (
            pl.concat(lazyframes, how="vertical")
            .collect()
            .unique(subset="video_id", keep="first")
            .filter(~pl.col("video_id").is_in(processed_video_ids))
        )
        if df.is_empty():
            logging.info("'%s' is up-to-date.", PathConfig.PROCESSED_DATA_PATH)
        else:
            # get the embedding model
            logging.info("Loading the '%s' text embedding model.", embedding_model_id)
            embedding_model: SentenceTransformer = SentenceTransformer(
                model_name_or_path=embedding_model_id, trust_remote_code=True
            )

            # generate serialized embeddings for the new records, ...
            # vertically concatenate them with the existing records, and ...
            # write the resulting pl.DataFrame to ~/data/youtube_transcripts.parquet
            logging.info(
                "Generating serialized embeddings for %s new YouTube video transcripts.",
                df.shape[0],
            )
            (
                pl.concat(
                    (
                        latest_data,
                        (
                            df.pipe(embed_transcripts, embedding_model)
                            .with_columns(pl.col("creation_date").str.to_datetime())
                            .select(latest_data.columns)
                        ),
                    ),
                    how="vertical",
                )
                .sort(by="creation_date")
                .write_parquet(PathConfig.PROCESSED_DATA_PATH)
            )
            logging.info(
                "Finished! '%s' has been updated with the new YouTube video transcripts.",
                PathConfig.PROCESSED_DATA_PATH,
            )
    except Exception as e:
        raise e


if __name__ == "__main__":
    channel_ids: list[str] = load_config().get("youtube_channel_ids")
    model_id: str = load_config().get("embedding_model_id")
    etl(channel_ids, model_id)
