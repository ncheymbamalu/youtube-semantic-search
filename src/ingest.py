import json
import os

from pathlib import PosixPath

import polars as pl
import requests

from dotenv import load_dotenv
from requests import Response
from pytube import Channel
from youtube_transcript_api import YouTubeTranscriptApi

from src.config import Config, load_config
from src.logger import logging

load_dotenv()


# NOTE: ./.venv/lib/python3.10/site-packages/pytube/extract.py needs to be modified, so ...
# that the 'pytube' library's 'Channel' class can properly read the YouTube video URLs ...
# that are listed in ./config.yaml
# source: https://stackoverflow.com/questions/74957606/pytube-and-the-new-channel-urls


def etl(youtube_channel_id: str, max_results: int = 50) -> pl.LazyFrame:
    """
    Extracts the ID, creation datetime, title, and transcription of several
    YouTube videos, and returns a pl.LazyFrame containing the extracted data.
    NOTE: the following link, https://www.youtube.com/watch?v=qPKmPaNaCmE&t=1s,
    shows how to find any YouTube channel ID

    Args:
        youtube_channel_id (str): Unique ID for the YouTube channel of interest
        max_results (int): Maximum number of transcribed YouTube videos that will
        be extracted. Defaults to 50, which is the maximum number of results allowed
        by the YouTube API.
    """
    try:
        logging.info(
            "Extracting and transcribing video data from the YouTube channel ID, '%s'.",
            youtube_channel_id
        )
        params: dict[str, int | list[str] | str] = {
            "key": os.getenv("YOUTUBE_API_KEY"),
            "channelId": youtube_channel_id,
            "part": ["snippet", "id"],
            "order": "date",
            "maxResults": max_results
        }
        response: Response = requests.get(
            "https://www.googleapis.com/youtube/v3/search", params=params
        )
        video_records: list[dict[str, str]] = []
        for item in json.loads(response.text).get("items"):
            try:
                video_record: dict[str, str] = {
                    "video_id": item.get("id").get("videoId"),
                    "datetime": item.get("snippet").get("publishedAt"),
                    "title": item.get("snippet").get("title"),
                    "transcript": " ".join(
                        transcript_dict.get("text") for transcript_dict in
                        YouTubeTranscriptApi.get_transcript(item.get("id").get("videoId"))
                    )
                }
                video_records.append(video_record)
            except:
                logging.info(
                    "Video ID, '%s', doesn't have a transcript.", item.get("id").get("videoId")
                )
        return (
            pl.LazyFrame(video_records)
            .with_columns(
                pl.col("datetime").cast(pl.Datetime),
                pl.col("title").str.replace_many(["&#39;", "&amp;", "  "], ["'", "&", " "]),
                pl.col("transcript").str.replace_many(["&#39;", "&amp;", "  "], ["'", "&", " "])
            )
            .unique(subset="video_id")
            .sort(by="datetime")
        )
    except Exception as e:
        raise e


def main() -> None:
    # create the 'data' sub-directory
    output_dir: PosixPath = Config.Path.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # a list of YouTube channel IDs
    youtube_channel_ids: list[str] = [
        Channel(url).channel_id for url in load_config().youtube_channels
    ]
    
    # extract and transcribe video data from each YouTube channel ID
    # write the transcribed video data to ./data/video_transcripts.parquet
    (
        pl.concat([etl(channel_id) for channel_id in youtube_channel_ids])
        .sort("datetime")
        .collect()
        .write_parquet(output_dir / "video_transcripts.parquet")
    )


if __name__ == "__main__":
    main()
