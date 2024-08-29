from gradio import Blocks, Button, Row, Textbox
from gradio.components import HTML, Markdown

from src.etl import latest_data
from src.semantic_search import get_indices


def format_result(
    video_id: str, title: str, width: int = 600, height: int = 300
) -> tuple[str, str]:
    markdown_text: str = f"""
    ## {title}<br>
    **[Video Link](https://youtu.be/{video_id})**
    """
    url: str = f"https://www.youtube.com/embed/{video_id}"
    embedded_video: str = f'<iframe width="{width}" height="{height}" src="{url}"></iframe>'
    return markdown_text, embedded_video


def query_results(query: str, cols: list[str] | None = None) -> list[HTML | Markdown]:
    if cols is None:
        cols = ["video_id", "title"]

    # get the top k indices
    idx: list[int] = get_indices(query)

    # extract the top k video IDs and their corresponding titles, as a dictionary
    response: dict[str, list[str]] = latest_data[idx, cols].to_dict(as_series=False)

    # create the gradio UI results
    results: list[HTML | Markdown] = []
    for i in range(len(idx)):
        video_id: str = response.get(cols[0])[i]
        title: str = response.get(cols[1])[i]
        markdown_text, embedded_video = format_result(video_id, title)
        results += [
            HTML(value=embedded_video, visible=True),
            Markdown(value=markdown_text, visible=True),
        ]
    return results


def display_results() -> None:
    k: int = 5
    outputs: list[HTML | Markdown] = []
    with Blocks() as demo:
        Markdown("## YouTube Search API")

        with Row():
            query: Textbox = Textbox(placeholder="Enter query", label="", scale=8)
            Button("Submit", size="sm").click(fn=query_results, inputs=query, outputs=outputs)

        for _ in range(k):
            with Row():
                outputs += [HTML(), Markdown()]
        query.submit(fn=query_results, inputs=query, outputs=outputs)
    demo.launch(debug=True, share=True)


if __name__ == "__main__":
    display_results()
