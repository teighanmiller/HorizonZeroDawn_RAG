from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, collection_name: str, dense_model: SentenceTransformer):
        self.qclient = QdrantClient("http://localhost:6333", timeout=60)
        self.collection_name = collection_name
        self.dense_model = dense_model

    def retrieval(self, query: str, classification: str, limit: int = 3):
        rag_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="classification",
                    match=models.MatchValue(value=classification),
                )
            ]
        )

        dense_vector = self.dense_model.encode(query)
        result = self.qclient.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="nomic",
            query_filter=rag_filter,
            with_payload=True,
        )

        return result.points
