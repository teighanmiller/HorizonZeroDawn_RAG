import uuid
import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from scraper import scrape_data

DENSE_DIM = 768
BATCH_SIZE = 50
COLLECTION_NAME = "HORIZON_RAG"


def get_id(row: pd.Series) -> str:
    row = row.fillna("")
    unique_id = uuid.uuid5(
        uuid.NAMESPACE_DNS,
        str(row["url"])
        + str(row["classification"])
        + str(row["category"])
        + str(row["location"])
        + row["content"],
    )
    return str(unique_id)


def featurize_data(
    df: pd.DataFrame, embedding_model: SentenceTransformer
) -> pd.DataFrame:
    df["nomic_vector"] = df["content"].apply(
        lambda x: embedding_model.encode(x if isinstance(x, str) else "Empty")
    )

    df["id"] = df.apply(get_id, axis=1)

    return df


def create_collection(client: QdrantClient, collection_data: pd.DataFrame):
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "nomic": models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    rows = [row for _, row in collection_data.iterrows()]

    for index in range(0, len(rows), BATCH_SIZE):
        batch = rows[index : index + BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=row["id"],
                    vector={
                        "nomic": row["nomic_vector"],
                        "bm25": models.Document(
                            text=(
                                row["content"]
                                if isinstance(row["content"], str)
                                else "Empty"
                            ),
                            model="Qdrant/bm25",
                        ),
                    },
                    payload={
                        "url": row["url"],
                        "classification": row["classification"],
                        "content": (
                            row["content"]
                            if isinstance(row["content"], str)
                            else "Empty"
                        ),
                        "category": (
                            row["category"]
                            if isinstance(row["category"], str)
                            else "None"
                        ),
                        "location": (
                            row["location"]
                            if isinstance(row["location"], str)
                            else "None"
                        ),
                    },
                )
                for row in batch
            ],
        )


def ingest():
    dense_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    qclient = QdrantClient("http://localhost:6333", timeout=60)

    raw_data_path = scrape_data()
    raw_data = pd.read_csv(raw_data_path)
    embedded_data = featurize_data(raw_data, dense_model)
    create_collection(client=qclient, collection_data=embedded_data)
