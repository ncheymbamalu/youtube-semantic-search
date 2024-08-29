from fastapi import FastAPI

from src.etl import latest_data
from src.semantic_search import get_indices

# create the 'FastAPI' instance
app: FastAPI = FastAPI(title="YouTube Search API")


# create the '/search' endpoint
@app.get("/search", response_model=list[dict[str, str]])
def search(query: str):
    idx: list[int] = get_indices(query)
    return [
        {"Title": title, "URL": f"https://youtu.be/{video_id}"}
        for title, video_id in zip(latest_data[idx, "title"], latest_data[idx, "video_id"])
    ]
