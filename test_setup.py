from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult

docs = [
    SearchResult(docid=i, text=f"this is passage {i}", score=None) for i in range(100)
]
query = "Give me passage 34"

ranker = SetwiseLlmRanker(
    model_name_or_path="google/flan-t5-large",
    tokenizer_name_or_path="google/flan-t5-large",
    device="cuda",
    num_child=10,
    scoring="generation",
    method="heapsort",
    k=10,
)

print(ranker.rerank(query, docs)[0])
