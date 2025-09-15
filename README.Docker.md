# HorizonZeroDawn_RAG - Docker Setup

This project demonstrates running the HorizonZeroDawn_RAG system using Docker and Docker Compose.  
It includes:

- A Streamlit app frontend
- Qdrant vector database for embeddings
- Data ingestion scripts

## Prerequisites

- Docker >= 24.x
- Docker Compose >= 2.x
- `git` and `curl` installed locally

## Docker Compose Services

| Service            | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `qdrant`           | Vector database used for storing embeddings.                     |
| `ingest`           | Script to ingest new data into Qdrant.                           |
| `ingest_preloaded` | Script to ingest preloaded embeddings/models for faster startup. |
| `app`              | Streamlit frontend for interacting with the RAG system.          |

## Environment Variables

The project uses an `.env` file for configuration. Key variables:
AZURE_OPENAI_ENDPOINT_URI
AZURE_OPENAI_API_KEY
AZURE_API_VERSION

```bash
HF_HOME=/root/.cache/huggingface  # Hugging Face cache for models
```

## Running the Project

### 1. Build all Docker images

```bash
docker compose build
```

### 2. Start Qdrant image

```bash
docker compose up -d qdrant
```

### 3. Ingest Preloaded Data

```bash
docker compose run ingest_preloaded
```

**_Note: If you want to scrape new data you can run the following command instead:_**

```bash
docker compose run ingest
```

**_This is will take several hours_**

### 4. Start the app

```bash
docker compose up app
```

### 5. Access the app

- In a browser access the app via http://localhost:8501

### 6. Use the app.

- Ask a question like "Who is Aloy?" or "Describe the weaknesses of a Thunderjaw."

## Stoping containers and cleaning

- To stop containers run:

```bash
docker compose down
```

- To clean container if you see warnings you can run commands with --remove-orphans, as shown in the following example:

```bash
docker compose up --remove-orphans
```

---

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
