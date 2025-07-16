import chromadb

class RetrievalDB:
    def __init__(self, prompts, embeddings, solutions, collection_name="humaneval"):
        self.client = chromadb.Client()
        # Check if collection exists
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            # If not, create it and populate
            self.collection = self.client.create_collection(name=collection_name)
            for idx, (emb, prompt, solution) in enumerate(zip(embeddings, prompts, solutions)):
                self.collection.add(
                    ids=[str(idx)],
                    embeddings=[emb.tolist()],
                    metadatas=[{"prompt": prompt, "solution": solution}]
                )

    def retrieve_similar_context(self, query_emb, k=1):
        results = self.collection.query(query_embeddings=[query_emb], n_results=k)
        return results["metadatas"] 
