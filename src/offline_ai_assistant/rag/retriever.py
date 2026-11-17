
import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from offline_ai_assistant.rag.vector_store import VectorStore


class Retriever:
    """
    Retrieve relevant document chunks from FAISS.

    Features:
    1. Normal single-query retrieval.
    2. Category-aware multi-query retrieval.
    3. Reciprocal Rank Fusion (RRF).
    4. Duplicate chunk removal.
    5. Category coverage for broad benefit questions.
    6. Focused retrieval for specific benefit questions.
    """

    RRF_K = 60

    # Each category has its own focused search query.
    # Medical insurance and digital healthcare are separate
    # because they are different employee benefits.

    BENEFIT_CATEGORY_QUERIES = {
        "medical_insurance": (
            "Share India employee medical insurance coverage "
            "medical insurance policy for employees"
        ),

        "digital_healthcare": (
            "Share India free digital healthcare facility "
            "Practo online doctor teleconsultation"
        ),

        "meals": (
            "Share India subsidized employee meals "
            "cafeteria breakfast lunch and snacks"
        ),

        "transport": (
            "Share India employee free transport facility "
            "fixed routes office commute"
        ),

        "outings": (
            "Share India corporate fun outings "
            "employee team outings activities"
        ),

        "engagement": (
            "Share India employee engagement activities "
            "birthday celebrations and events"
        ),

        "yoga_wellness": (
            "Share India free weekend online yoga sessions "
            "employee wellness benefits"
        ),
    }

    BENEFIT_TERMS = {
        "benefit",
        "benefits",
        "facility",
        "facilities",
        "perks",
        "welfare",
    }

    # Exact insurance-detail questions need a narrower retrieval path.
    # These patterns target identifiers/amounts, not ordinary insurance
    # benefit questions such as "Does the company provide insurance?".
    EXACT_INSURANCE_DETAIL_PATTERNS = (
        re.compile(r"\bpolicy\s*(?:number|no\.?)\b", re.IGNORECASE),
        re.compile(r"\binsurance\s*(?:number|id|identifier)\b", re.IGNORECASE),
        re.compile(r"\b(?:coverage|cover)\s*(?:amount|limit)\b", re.IGNORECASE),
        re.compile(r"\bsum\s*insured\b", re.IGNORECASE),
        re.compile(r"\b(?:policy|insurance)\s*(?:expiry|expiration)\b", re.IGNORECASE),
        re.compile(r"\b(?:expiry|expiration)\s*date\b", re.IGNORECASE),
    )

    CATEGORY_TERMS = {
        "medical_insurance": {
            "medical",
            "insurance",
        },

        "digital_healthcare": {
            "healthcare",
            "teleconsultation",
            "teleconsultations",
            "practo",
            "doctor",
            "doctors",
            "digital",
        },

        "meals": {
            "meal",
            "meals",
            "food",
            "cafeteria",
            "breakfast",
            "lunch",
            "snacks",
        },

        "transport": {
            "transport",
            "transportation",
            "commute",
            "commuting",
        },

        "outings": {
            "outing",
            "outings",
        },

        "engagement": {
            "engagement",
            "celebration",
            "celebrations",
            "events",
            "activities",
        },

        "yoga_wellness": {
            "yoga",
            "wellness",
            "fitness",
            "wellbeing",
        },
    }

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 3,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.vector_store = vector_store
        self.top_k = top_k

    @staticmethod
    def _tokenize(query: str) -> set:
        """Normalize query into individual words."""

        return set(
            re.findall(r"\b\w+\b", query.lower())
        )

    def _get_matched_categories(
        self,
        query: str,
    ) -> List[str]:
        """Identify categories explicitly mentioned in a query."""

        tokens = self._tokenize(query)

        return [
            category
            for category, terms in self.CATEGORY_TERMS.items()
            if tokens.intersection(terms)
        ]

    def is_broad_benefits_query(
        self,
        query: str,
    ) -> bool:
        """Identify general and multi-category benefit questions."""

        tokens = self._tokenize(query)

        has_benefit_term = bool(
            tokens.intersection(self.BENEFIT_TERMS)
        )

        matched_categories = self._get_matched_categories(
            query
        )

        # Multiple categories indicate a broad question.
        if len(matched_categories) >= 2:
            return True

        # General benefits question without a specific category.
        if has_benefit_term and not matched_categories:
            return True

        return False

    def _is_exact_insurance_detail_query(
        self,
        query: str,
    ) -> bool:
        """Identify requests for exact insurance identifiers or values."""

        if "medical_insurance" not in self._get_matched_categories(query):
            return False

        return any(
            pattern.search(query)
            for pattern in self.EXACT_INSURANCE_DETAIL_PATTERNS
        )

    def _build_search_queries(
        self,
        query: str,
    ) -> List[str]:
        """
        Build original and focused search queries.

        Specific questions:
            Original question + matched category query.

        Broad questions:
            Original question + all relevant category queries.
        """

        matched_categories = self._get_matched_categories(
            query
        )

        # General benefits questions search every category.
        if self.is_broad_benefits_query(query):

            if not matched_categories:
                matched_categories = list(
                    self.BENEFIT_CATEGORY_QUERIES.keys()
                )

        # Specific questions without a recognized category
        # continue using normal single-query retrieval.
        elif not matched_categories:
            return [query]

        queries = [query]

        for category in matched_categories:
            queries.append(
                self.BENEFIT_CATEGORY_QUERIES[category]
            )

        return list(dict.fromkeys(queries))

    @staticmethod
    def _get_document_key(
        document: Document,
    ) -> tuple:
        """Generate a unique key for duplicate detection."""

        metadata = document.metadata

        content = " ".join(
            document.page_content.lower().split()
        )

        return (
            metadata.get("source", ""),
            metadata.get(
                "page_label",
                metadata.get("page", ""),
            ),
            content,
        )

    def _reciprocal_rank_fusion(
        self,
        ranked_results: List[List[Tuple[Document, float]]],
        k: int,
    ) -> List[Tuple[Document, float]]:
        """Merge ranked results using Reciprocal Rank Fusion."""

        fused_scores: Dict[tuple, float] = {}
        documents: Dict[tuple, Document] = {}

        for results in ranked_results:

            for rank, (document, _distance) in enumerate(
                results,
                start=1,
            ):
                document_key = self._get_document_key(
                    document
                )

                if document_key not in fused_scores:
                    fused_scores[document_key] = 0.0
                    documents[document_key] = document

                fused_scores[document_key] += (
                    1.0 / (self.RRF_K + rank)
                )

        sorted_keys = sorted(
            fused_scores,
            key=lambda key: fused_scores[key],
            reverse=True,
        )

        return [
            (
                documents[key],
                float(fused_scores[key]),
            )
            for key in sorted_keys[:k]
        ]

    def _retrieve_with_category_coverage(
        self,
        query: str,
        categories: List[str],
        k: int,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve one leading document from every requested
        benefit category before applying a global cutoff.

        This prevents one category from dominating all results.
        """

        candidate_k = max(k, 5)

        category_results = []

        for category in categories:

            search_query = self.BENEFIT_CATEGORY_QUERIES[
                category
            ]

            results = (
                self.vector_store.similarity_search_with_score(
                    query=search_query,
                    k=candidate_k,
                )
            )

            category_results.append(results)

        # Calculate combined RRF scores.
        fused_results = self._reciprocal_rank_fusion(
            ranked_results=category_results,
            k=max(k, len(categories) * candidate_k),
        )

        fused_score_map = {
            self._get_document_key(document): score
            for document, score in fused_results
        }

        selected = []
        selected_keys = set()

        # First, preserve category coverage.
        for results in category_results:

            if not results:
                continue

            document, _distance = results[0]

            document_key = self._get_document_key(
                document
            )

            if document_key in selected_keys:
                continue

            selected_keys.add(document_key)

            score = fused_score_map.get(
                document_key,
                0.0,
            )

            selected.append(
                (document, score)
            )

            if len(selected) >= k:
                return selected

        return selected

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Document]:
        """Retrieve documents without scores."""

        results = self.retrieve_with_scores(
            query=query,
            k=k,
        )

        return [
            document
            for document, _score in results
        ]

    def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using normal, focused,
        or category-aware search.

        Specific benefit questions:
            Original query + focused category query.

        Broad questions:
            Category-wise retrieval with coverage.

        Other questions:
            Normal FAISS retrieval.

        Returned scores are RRF scores, not FAISS distances.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        final_k = (
            k if k is not None else self.top_k
        )

        if final_k <= 0:
            raise ValueError("k must be positive.")

        # --------------------------------------------------
        # EXACT INSURANCE DETAIL LOOKUP
        # --------------------------------------------------

        if self._is_exact_insurance_detail_query(query):
            # For exact identifiers (for example, a policy number),
            # use the original wording and pass only its strongest
            # direct match to the answer generator. This avoids
            # adding unrelated documents from general category search.
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=1,
            )

            return self._reciprocal_rank_fusion(
                ranked_results=[results],
                k=1,
            )

        # --------------------------------------------------
        # BROAD BENEFIT QUESTION RETRIEVAL
        # --------------------------------------------------

        if self.is_broad_benefits_query(query):

            categories = self._get_matched_categories(
                query
            )

            # General benefits questions search every category.
            if not categories:
                categories = list(
                    self.BENEFIT_CATEGORY_QUERIES.keys()
                )

            return self._retrieve_with_category_coverage(
                query=query,
                categories=categories,
                k=final_k,
            )

        # --------------------------------------------------
        # SPECIFIC QUESTION RETRIEVAL
        # --------------------------------------------------

        search_queries = self._build_search_queries(
            query
        )

        # No recognized benefit category:
        # Preserve normal FAISS retrieval.
        if len(search_queries) == 1:

            results = (
                self.vector_store.similarity_search_with_score(
                    query=query,
                    k=final_k,
                )
            )

            return self._reciprocal_rank_fusion(
                ranked_results=[results],
                k=final_k,
            )

        # Retrieve candidates from both original and focused
        # searches. The final output still respects final_k.
        candidate_k = max(final_k, 5)

        ranked_results = []

        for search_query in search_queries:

            results = (
                self.vector_store.similarity_search_with_score(
                    query=search_query,
                    k=candidate_k,
                )
            )

            ranked_results.append(results)

        # Combine rankings. Documents appearing near the top
        # in multiple searches receive stronger RRF scores.
        return self._reciprocal_rank_fusion(
            ranked_results=ranked_results,
            k=final_k,
        )
