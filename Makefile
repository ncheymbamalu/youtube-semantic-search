.PHONY: etl backend frontend

# Data extraction, transformation, and loading pipeline
etl:
	poetry run python src/etl.py

# FastAPI application
backend:
	uvicorn src.app:app --reload

# Gradio application
frontend:
	poetry run python src/frontend.py
