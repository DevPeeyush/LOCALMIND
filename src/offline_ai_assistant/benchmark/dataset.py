
"""Default evaluation dataset for the Offline AI Assistant."""

BENCHMARK_DATASET = [
    {
        "id": "general_001",
        "category": "General Knowledge",
        "question": "What is artificial intelligence?",
        "expected_answer": (
            "Artificial intelligence is the ability of machines "
            "to perform tasks that normally require human intelligence."
        ),
    },
    {
        "id": "general_002",
        "category": "General Knowledge",
        "question": "What is machine learning?",
        "expected_answer": (
            "Machine learning is a branch of AI where systems learn "
            "patterns from data to make predictions or decisions."
        ),
    },
    {
        "id": "general_003",
        "category": "General Knowledge",
        "question": "What is retrieval augmented generation?",
        "expected_answer": (
            "RAG combines information retrieval with language "
            "generation to produce answers grounded in retrieved data."
        ),
    },
    {
        "id": "reasoning_001",
        "category": "Reasoning",
        "question": (
            "A person has 3 boxes with 4 apples in each box. "
            "How many apples are there in total?"
        ),
        "expected_answer": "There are 12 apples in total.",
    },
    {
        "id": "reasoning_002",
        "category": "Reasoning",
        "question": (
            "If a task takes 10 minutes and is repeated 3 times, "
            "how much total time is required?"
        ),
        "expected_answer": "30 minutes.",
    },
    {
        "id": "coding_001",
        "category": "Coding",
        "question": "Write a Python function to add two numbers.",
        "expected_answer": (
            "A Python function that accepts two numbers "
            "and returns their sum."
        ),
    },
    {
        "id": "summarization_001",
        "category": "Summarization",
        "question": (
            "Summarize this statement: RAG retrieves relevant "
            "documents and provides their context to an LLM "
            "before generating an answer."
        ),
        "expected_answer": (
            "RAG retrieves relevant information and gives it "
            "to an LLM to generate a context-based answer."
        ),
    },
    {
        "id": "instruction_001",
        "category": "Instruction Following",
        "question": (
            "Explain Python in exactly one sentence."
        ),
        "expected_answer": (
            "Python is a programming language known for its "
            "readability and versatility."
        ),
    },
]


def get_benchmark_dataset():
    """Return a copy of the default benchmark dataset."""
    return [item.copy() for item in BENCHMARK_DATASET]
