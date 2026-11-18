
"""Answer quality evaluation for model benchmarking."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_answer(
    generated_answer: str,
    expected_answer: str,
) -> dict:
    """Evaluate an answer using TF-IDF cosine similarity."""

    if not generated_answer or not expected_answer:
        return {
            "similarity_score": 0.0,
            "quality": "Poor",
        }

    vectorizer = TfidfVectorizer()

    try:
        vectors = vectorizer.fit_transform(
            [generated_answer, expected_answer]
        )

        score = float(
            cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        )

    except ValueError:
        score = 0.0

    if score >= 0.70:
        quality = "Good"
    elif score >= 0.40:
        quality = "Average"
    else:
        quality = "Poor"

    return {
        "similarity_score": round(score * 100, 2),
        "quality": quality,
    }
