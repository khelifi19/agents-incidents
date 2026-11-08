FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY data/ data/
ENV OLLAMA_HOST=http://host.docker.internal:11434
EXPOSE 8000
CMD ["uvicorn", "app.api:api", "--host", "0.0.0.0", "--port", "8000"]