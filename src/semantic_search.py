import gradio as gr
import numpy as np
import polars as pl

from gradio.components import HTML, Markdown as GradioMarkdown, Textbox
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config


# read in ../data/transcript_embeddings.parquet as a pl.LazyFrame
df: pl.LazyFrame = pl.scan_parquet(Config.Path.DATA_DIR / "transcript_embeddings.parquet")

# load the embedding model from ../artifacts/embedding_model/
model: SentenceTransformer = SentenceTransformer(str(Config.Path.ARTIFACTS_DIR / "embedding_model"))


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


def format_result(
        video_id: str, title: str, width: int = 600, height: int = 300
) -> tuple[str, str]:
    markdown_text: str = f"""
    ## {title}<br>
    **[Video Link](https://youtu.be/{video_id})**
    """
    uri: str = f"https://www.youtube.com/embed/{video_id}"
    embedded_video: str = f'<iframe width="{width}" height="{height}" src="{uri}"></iframe>'
    return markdown_text, embedded_video


def get_search_results(
        query: str, cols: list[str] = ["video_id", "title"]
) -> list[HTML | GradioMarkdown]:
    # get the top k indices
    indices: list[int] = get_indices(query)

    # extract the top k video IDs and their corresponding titles, as a dictionary
    response: dict[str, list[str]] = df.select(cols).collect()[indices].to_dict(as_series=False)

    # create the gradio UI results
    results: list[HTML | GradioMarkdown] = []
    for idx in range(len(indices)):
        video_id: str = response.get(cols[0])[idx]
        title: str = response.get(cols[1])[idx]
        markdown_text, embedded_video = format_result(video_id, title)
        results += [
            gr.HTML(value=embedded_video, visible=True),
            gr.Markdown(value=markdown_text, visible=True)
        ]
    return results


def main() -> None:
    k: int = 5
    outputs: list[HTML | GradioMarkdown] = []
    with gr.Blocks() as demo:
        gr.Markdown("## YouTube Search API")

        with gr.Row():
            query: Textbox = gr.Textbox(placeholder="Enter query", label="", scale=8)
            gr.Button("Submit", size="sm").click(
                fn=get_search_results, inputs=query, outputs=outputs
            )

        for _ in range(k):
            with gr.Row():
                outputs += [gr.HTML(), gr.Markdown()]
        query.submit(fn=get_search_results, inputs=query, outputs=outputs)
    demo.launch(debug=True, share=True)


if __name__ == "__main__":
    main()
