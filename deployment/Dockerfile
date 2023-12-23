#In this docker file, we build two images (docker multi-stage builds), one to install poetry in and generate a requirements.txt file,
#and the other to install the requirements.txt file and run the web service. This reduces by half the size of the image.
# tutorial: https://denisbrogg.hashnode.dev/efficient-python-docker-image-from-any-poetry-project

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
COPY deployment/deploy.py /app/deploy.py
COPY deployment/inference.py /app/inference.py
COPY --from=builder /app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
