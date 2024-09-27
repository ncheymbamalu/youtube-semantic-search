import json
import os

import pandas as pd
import polars as pl
import requests

from dotenv import load_dotenv
from requests import Response
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast
from youtube_transcript_api import YouTubeTranscriptApi

from src.logger import logging
from src.paths import PathConfig

load_dotenv(PathConfig.PROJECT_DIR / ".env")


def transcribe_videos(youtube_channel_id: str, max_results: int = 50) -> pl.LazyFrame:
    """Returns a pl.DataFrame containing the video ID, creation date, title,
    and transcript, for at most, 50 videos, for a given YouTube channel ID
    NOTE: the following link, https://www.youtube.com/watch?v=qPKmPaNaCmE&t=1s,
    shows how to find the YouTube channel ID for a given YouTube channel

    Args:
        youtube_channel_id (str): ID of the YouTube channel whose videos will
        be transcribed.
        max_results (int, optional): Number of YouTube videos that will be transcribed.
        Defaults to 50.

    Returns:
        pl.LazyFrame: Dataset containing the YouTube channel ID's transcribed videos.
    """
    try:
        params: dict[str, int | list[str] | str] = {
            "key": os.getenv("YOUTUBE_API_KEY"),
            "channelId": youtube_channel_id,
            "part": ["snippet", "id"],
            "order": "date",
            "maxResults": max_results,
        }
        response: Response = requests.get(PathConfig.YOUTUBE_DATA_API, params=params)
        records: list[dict[str, str]] = []
        for item in json.loads(response.text).get("items"):
            try:
                record: dict[str, str] = {
                    "video_id": item.get("id").get("videoId"),
                    "creation_date": item.get("snippet").get("publishedAt"),
                    "title": item.get("snippet").get("title").strip(),
                    "transcript": " ".join(
                        transcript_dict.get("text").strip()
                        for transcript_dict in YouTubeTranscriptApi.get_transcript(
                            item.get("id").get("videoId")
                        )
                        if transcript_dict.get("text")
                    ),
                }
                records.append(record)
            except:
                logging.info(
                    "The video titled, '%s', doesn't have a transcript.",
                    item.get("snippet").get("title"),
                )
        unusual_strings: list[str] = ["&#39;", "&quot;", "&amp;", "  "]
        correct_strings: list[str] = ["'", "'", "&", " "]
        return pl.LazyFrame(records).with_columns(
            pl.col("creation_date").str.to_datetime(),
            pl.col("title").str.replace_many(unusual_strings, correct_strings),
            pl.col("transcript").str.replace_many(unusual_strings, correct_strings),
        )
    except Exception as e:
        raise e


def chunk_text(document: str, tokenizer: BertTokenizerFast, token_overlap: int = 50) -> list[str]:
    """This function takes in as input a document, i.e., string of text, and returns
    a list containing the document, if its number of tokens is less than or equal to
    the tokenizer's context length. However, if its number of tokens is greater than
    the tokenizer's context length, the document is split into smaller documents, i.e.,
    chunks, where the number of tokens for each smaller document is less than or equal
    to the tokenizer's context length.  The smaller documents are then stored as a list,
    and returned.

    Args:
        document (str): String of text
        tokenizer (BertTokenizerFast): Object that converts a string of text to tokens
        token_overlap (int, optional): Number of overlapping tokens between successive
        chunks. Defaults to 50.

    Returns:
        list[str]: List containing the original document if its number of tokens <= the
        tokenizer's context length, or a list containing smaller documents (chunks), if
        the original document's number of tokens > the tokenizer's context length.
    """
    try:
        max_length: int = tokenizer.model_max_length
        tokens: list[str] = tokenizer.tokenize(document)
        chunks: list[str] = []
        for i in range(0, len(tokens), max_length - token_overlap):
            chunk_tokens: list[str] = tokens[i : i + max_length]
            chunk: str = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk)
        return chunks
    except Exception as e:
        raise e


def embed_transcripts(data: pl.DataFrame, model: SentenceTransformer) -> pl.DataFrame:
    """Generates serialized embeddings, i.e., bytes, for each record's transcripts.
    NOTE: if a single transcript's context length (number of tokens) is greater than
    the embedding model's tokenizer's context length, it will be mapped to more than
    one embedding vector.

    Args:
        data (pl.DataFrame): Dataset containing the video ID, creation date, title, and
        transcript of each record (YouTube video)
        model (SentenceTransformer): Object that converts a transcript (string of text)
        to embeddings (1-D array of floating point numbers)

    Returns:
        pl.DataFrame: Dataset containing the video ID, creation date, title, transcript,
        and transcript embeddings (serialized bytes) for each record (YouTube video)
    """
    try:
        records: list[list[str]] = []
        for i in range(data.shape[0]):
            record: list[list[str]] = [
                data[i].transpose().to_series().to_list() + [model.encode(chunk).tobytes()]
                for chunk in chunk_text(data[i, "transcript"], model.tokenizer)
            ]
            records += record
        df: pd.DataFrame = pd.DataFrame(records, columns=data.columns + ["embedding"])
        return pl.from_pandas(df)
    except Exception as e:
        raise e
