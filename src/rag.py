from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class RAG:
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

        result = self.qclient.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=self.dense_model.encode(query), using="nomic", limit=limit
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using="bm25",
                    limit=limit,
                ),
            ],
            query_filter=rag_filter,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
        )

        return result.points[0]
