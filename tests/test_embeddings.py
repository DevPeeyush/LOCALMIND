
from offline_ai_assistant.rag.embeddings import EmbeddingService


def test_embeddings():

    embedding_service = EmbeddingService()

    texts = [
        "Python is a programming language.",
        "Machine Learning is a branch of AI.",
        "Deep Learning uses neural networks."
    ]

    embeddings = embedding_service.embed_documents(texts)

    print("\nEmbedding Results")

    for index, embedding in enumerate(embeddings):

        print(f"\nDocument {index + 1}")
        print("Vector dimensions:", len(embedding))
        print("First 5 values:", embedding[:5])

    query = "What is Machine Learning?"

    query_embedding = embedding_service.embed_query(query)

    print("\nQuery Embedding")
    print("Vector dimensions:", len(query_embedding))

        # Validate results
    assert len(embeddings) == 3
    
    embedding_dimension = len(embeddings[0])
    
    assert all(
        len(vector) == embedding_dimension
        for vector in embeddings
    )
    
    assert len(query_embedding) == embedding_dimension
    
    print(f"Embedding dimensions: {embedding_dimension}")
    print("\nEmbedding generation successful!")
    
    
if __name__ == "__main__":
    test_embeddings()
