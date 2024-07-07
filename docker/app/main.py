from fastapi import FastAPI

from app.utils import df, get_indices


# instantiate an object of type, 'FastAPI'
app: FastAPI = FastAPI(title="YouTube Search API")

# FastAPI operations
@app.get("/", response_model=dict[str, str])
def health_check():
    return {"health_check": "OK"}


@app.get("/search", response_model=dict[str, list[str]])
def search(query: str):
    indices: list[int] = get_indices(query)
    return df.select(["video_id", "title"]).collect()[indices].to_dict(as_series=False)
