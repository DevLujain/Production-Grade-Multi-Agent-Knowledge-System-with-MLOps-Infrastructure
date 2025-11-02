FROM python:3.10

WORKDIR /app
ENV CHROMADB_PERSIST_DIR=/app/data/vectordb

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/vectordb
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "3000"]
