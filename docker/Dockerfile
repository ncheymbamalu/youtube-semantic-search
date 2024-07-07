FROM python:3.10

RUN apt-get -qq update \
    && apt-get -qq -y install curl tree \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  DEBIAN_FRONTEND=noninteractive \
  PIP_NO_CACHE_DIR=off \
  VIRTUAL_ENV=/youtube_semantic_search/.venv \
  PATH="/youtube_semantic_search/.venv/bin:${PATH}"

RUN mkdir youtube_semantic_search \
    && python3 -m venv ${VIRTUAL_ENV} \
    && . ${VIRTUAL_ENV}/bin/activate \
    && pip install -U pip

WORKDIR /youtube_semantic_search

COPY . ./

RUN pip install -q -r requirements.txt --progress-bar off

EXPOSE 80

HEALTHCHECK CMD curl --fail https://localhost:80/_store/health

CMD ["fastapi", "run", "app/main.py", "--port", "80"]