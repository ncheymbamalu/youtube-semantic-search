.PHONY: etl

# Data extraction, transformation, and loading pipeline
etl:
	poetry run python src/etl.py