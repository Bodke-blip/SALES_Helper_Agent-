import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from qdrant_client import models


HYBRID_COLLECTION_NAME = os.getenv(
    "HYBRID_QDRANT_COLLECTION_NAME",
    "predikly_hybrid_search_data_v2",
)
HYBRID_FALLBACK_COLLECTION_NAME = os.getenv(
    "HYBRID_QDRANT_FALLBACK_COLLECTION_NAME",
    "predikly_hybrid_serch_data",
)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BM25_STATE_PATH = Path(os.getenv("BM25_STATE_PATH", "data/bm25_sparse_encoder.json"))
BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B = float(os.getenv("BM25_B", "0.75"))


def tokenize_for_sparse_vector(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1
    ]


class BM25SparseEncoder:
    def __init__(
        self,
        *,
        vocabulary: dict[str, int] | None = None,
        idf: dict[str, float] | None = None,
        avg_doc_len: float = 0.0,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.vocabulary = vocabulary or {}
        self.idf = idf or {}
        self.avg_doc_len = avg_doc_len
        self.k1 = k1
        self.b = b

    @classmethod
    def fit(cls, texts: list[str]) -> "BM25SparseEncoder":
        tokenized_documents = [tokenize_for_sparse_vector(text) for text in texts]
        document_count = len(tokenized_documents)
        document_frequencies: Counter[str] = Counter()

        for tokens in tokenized_documents:
            document_frequencies.update(set(tokens))

        vocabulary = {
            token: index
            for index, token in enumerate(sorted(document_frequencies))
        }
        idf = {
            token: math.log(1 + ((document_count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequencies.items()
        }
        avg_doc_len = (
            sum(len(tokens) for tokens in tokenized_documents) / document_count
            if document_count
            else 0.0
        )
        return cls(vocabulary=vocabulary, idf=idf, avg_doc_len=avg_doc_len)

    @classmethod
    def load(cls, path: Path = BM25_STATE_PATH) -> "BM25SparseEncoder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            vocabulary={str(key): int(value) for key, value in payload["vocabulary"].items()},
            idf={str(key): float(value) for key, value in payload["idf"].items()},
            avg_doc_len=float(payload["avg_doc_len"]),
            k1=float(payload.get("k1", BM25_K1)),
            b=float(payload.get("b", BM25_B)),
        )

    def save(self, path: Path = BM25_STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "vocabulary": self.vocabulary,
                    "idf": self.idf,
                    "avg_doc_len": self.avg_doc_len,
                    "k1": self.k1,
                    "b": self.b,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = tokenize_for_sparse_vector(text)
        token_counts = Counter(tokens)
        doc_len = len(tokens) or 1
        indices = []
        values = []

        for token, term_frequency in token_counts.items():
            if token not in self.vocabulary:
                continue

            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1))
            )
            score = self.idf.get(token, 0.0) * (
                (term_frequency * (self.k1 + 1)) / denominator
            )

            if score > 0:
                indices.append(self.vocabulary[token])
                values.append(float(score))

        return self._to_sparse_vector(indices, values)

    def encode_query(self, text: str) -> models.SparseVector:
        token_counts = Counter(tokenize_for_sparse_vector(text))
        indices = []
        values = []

        for token, term_frequency in token_counts.items():
            if token not in self.vocabulary:
                continue

            score = self.idf.get(token, 0.0) * term_frequency

            if score > 0:
                indices.append(self.vocabulary[token])
                values.append(float(score))

        return self._to_sparse_vector(indices, values)

    @staticmethod
    def _to_sparse_vector(indices: list[int], values: list[float]) -> models.SparseVector:
        pairs = sorted(zip(indices, values), key=lambda item: item[0])
        return models.SparseVector(
            indices=[index for index, _ in pairs],
            values=[value for _, value in pairs],
        )
