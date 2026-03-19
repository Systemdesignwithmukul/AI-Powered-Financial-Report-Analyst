from qdrant_client import QdrantClient

client  = QdrantClient(host="localhost", port=6333)
results = client.scroll(
    collection_name="financial_reports",
    limit=20,
    with_payload=True,
    with_vectors=False
)[0]

print("=== ALL CHUNKS IN QDRANT ===")
print(f"Total chunks found: {len(results)}\n")

for i, point in enumerate(results):
    text = point.payload.get("text", "")
    page = point.payload.get("page", "")
    print(f"Chunk {i+1} | Page {page}")
    print(text[:300])
    print("---")
