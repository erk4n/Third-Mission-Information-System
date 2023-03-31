# syntax=docker/dockerfile:1

FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir

COPY . .

CMD [ "uvicorn", "transfer_analyzer.api:app", "--host", "0.0.0.0", "--port", "8000"]

