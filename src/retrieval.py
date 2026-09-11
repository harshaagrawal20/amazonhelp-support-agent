"""
Historical Customer Support Retrieval Module for AmazonHelp.

Provides a lightweight, leakage-free TF-IDF retriever fit strictly on the
training partition (amazonhelp_dev_train.jsonl).
Given a customer inquiry, retrieves the top-k most lexically and semantically
similar historical customer messages alongside AmazonHelp's real resolution replies.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.intents import clean_tweet_text


class HistoricalRetriever:
    """
    Lightweight TF-IDF Historical Support Resolution Retriever.
    Strictly indexed on training interaction pairs to guarantee zero data leakage.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int = 30000,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.min_df = min_df
        self.records: List[Dict[str, Any]] = []
        self.tfidf_matrix: Optional[Any] = None
        self.is_fitted: bool = False

    def fit(self, records_or_path: Union[str, Path, List[Dict[str, Any]]]) -> "HistoricalRetriever":
        """
        Fit the retriever strictly on training interaction pairs.
        Accepts a filepath (JSONL) or a list of record dicts.
        """
        if isinstance(records_or_path, (str, Path)):
            path = Path(records_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Training dataset not found: {path}")
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            self.records = records
        else:
            self.records = list(records_or_path)

        if not self.records:
            raise ValueError("Cannot fit HistoricalRetriever on empty dataset.")

        # Adapt min_df if dataset is tiny (e.g. in unit tests)
        if len(self.records) < 10 and self.min_df > 1:
            self.vectorizer.set_params(min_df=1)
        else:
            self.vectorizer.set_params(min_df=self.min_df)

        texts = []
        for r in self.records:
            # Prefer cleaned_text if present, else clean customer_message
            if "cleaned_text" in r and r["cleaned_text"]:
                texts.append(r["cleaned_text"])
            else:
                raw_msg = r.get("customer_message") or r.get("customer_text") or ""
                texts.append(clean_tweet_text(raw_msg))

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return self

    def retrieve(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k historically resolved interaction pairs matching the query.
        Returns list of dicts with:
          - historical_customer_text
          - historical_amazonhelp_response
          - similarity_score
          - conversation_id
          - tweet_id
          - historical_intent
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("HistoricalRetriever must be fitted before calling retrieve().")

        if not query or not query.strip():
            return []

        cleaned_query = clean_tweet_text(query)
        if not cleaned_query:
            return []

        query_vec = self.vectorizer.transform([cleaned_query])
        if query_vec.nnz == 0:
            # Query has zero terms in the fitted vocabulary
            return []

        # Cosine similarity via dot product (TF-IDF vectors are L2-normalized by default in scikit-learn)
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

        top_k = min(k, len(self.records))
        if top_k <= 0:
            return []

        # Argpartition for fast top-k selection then sort descending
        partitioned_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = partitioned_indices[np.argsort(scores[partitioned_indices])[::-1]]

        results: List[Dict[str, Any]] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            r = self.records[idx]
            cust_text = r.get("customer_message") or r.get("customer_text") or ""
            resp_text = r.get("support_response") or r.get("support_text") or ""

            results.append({
                "historical_customer_text": cust_text,
                "historical_amazonhelp_response": resp_text,
                "similarity_score": round(score, 4),
                "conversation_id": r.get("conversation_id"),
                "tweet_id": r.get("customer_tweet_id") or r.get("tweet_id"),
                "historical_intent": r.get("intent"),
            })

        return results
