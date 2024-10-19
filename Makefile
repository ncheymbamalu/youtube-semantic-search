.PHONY: install check clean etl push backend frontend runner
.DEFAULT_GOAL:=runner

# Installs project dependencies
install: pyproject.toml
	poetry install

# Checks code via Ruff
check: install
	poetry run ruff check src

# Removes unwanted cache directories, all '__pycache__' directories and '~/.ruff_cache'
clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache

# Data extraction, transformation, and loading pipeline
etl:
	poetry run python src/etl.py


# Pushes the latest local data, ~/data/youtube_transcripts.parquet, to the remote DVC repo
push:
	dvc add ./data
	git add data.dvc
	git commit -m "updating ~/data/youtube_transcripts.parquet locally and pushing to remote"
	dvc push
	git push

# FastAPI application
backend:
	uvicorn src.app:app --reload

# Gradio application
frontend:
	poetry run python src/frontend.py

# ETL pipeline, removes unwanted directories, and pushes local data to the remote DVC repo
runner: etl clean push
