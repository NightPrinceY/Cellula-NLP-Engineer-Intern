import pandas as pd
from embedding import Embedder
from retrieval import RetrievalDB
from codegen import generate_code_with_context

class CodeGenPipeline:
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        self.prompts = self.df["prompt"].tolist()
        self.solutions = self.df["canonical_solution"].tolist()
        self.embedder = Embedder()
        self.embeddings = self.embedder.encode(self.prompts, batch_size=32, show_progress_bar=True)
        self.retrieval_db = RetrievalDB(self.prompts, self.embeddings, self.solutions)

    def generate_code_from_prompt(self, user_prompt, k=1):
        query_emb = self.embedder.encode([user_prompt])[0]
        retrieved = self.retrieval_db.retrieve_similar_context(query_emb, k=k)[0]
        context = "\n\n".join([f"# Task:\n{r['prompt']}\n{r['solution']}" for r in retrieved])
        return generate_code_with_context(user_prompt, context) 