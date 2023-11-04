#
# Build image
#

FROM python:3.10.6-slim AS builder

WORKDIR /app
COPY pyproject.toml /app
COPY poetry.lock /app

RUN pip install poetry
RUN poetry config virtualenvs.create false 
RUN poetry install --no-root --without dev
RUN poetry export -f requirements.txt >> requirements.txt

#
# Prod image
#

FROM python:3.10.6-slim AS runtime
WORKDIR /app
RUN mkdir /app/data/raw /app/data/interim /app/data/processed /app/models 
COPY src/* /app/src/
COPY training_job.py /app/training_job.py
COPY --from=builder /app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
