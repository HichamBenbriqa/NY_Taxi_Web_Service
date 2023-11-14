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

COPY src/ /app/src/
COPY config /app/config
COPY training_job.py /app/training_job.py
COPY .env /app/.env
COPY --from=builder /app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN dvc init --no-scm
RUN dvc remote add -d storage s3://mlops-nyc-taxi-project/web-service/
RUN python training_job.py
RUN dvc add /app/data/
RUN dvc push
