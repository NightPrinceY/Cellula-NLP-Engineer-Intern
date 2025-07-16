# Week 3: CodeGenBot – Retrieval-Augmented Code Generation Assistant

## Overview
This week’s project extends the Cellula AI internship into code intelligence by building **CodeGenBot**: a retrieval-augmented Python code generation assistant. The system leverages semantic search over the HumanEval dataset and a state-of-the-art LLM to generate Python code from user prompts, grounded in real coding examples. The app is delivered as a modular, production-ready Streamlit chatbot.

## Features
- **Retrieval-Augmented Generation (RAG):** Combines semantic search with LLM code synthesis for accurate, context-aware code generation.
- **Semantic Search:** Uses Sentence Transformers to embed and retrieve similar coding problems from HumanEval.
- **LLM Integration:** Utilizes DeepSeek-R1-Distill-Qwen-1.5B via HuggingFace Inference API for high-quality Python code generation.
- **Conversational UI:** Streamlit-based chat interface with code formatting, chat history, and error handling.
- **Modular Pipeline:** Clean separation of embedding, retrieval, and generation logic for easy extension and maintenance.

## Approach & Methodology
- **Dataset:** [HumanEval](https://huggingface.co/datasets/openai_humaneval) – real-world Python coding problems and solutions.
- **Embedding:** Prompts are embedded using `all-MiniLM-L6-v2` (Sentence Transformers) for semantic similarity search.
- **Retrieval:** Vector search retrieves the most relevant coding examples to provide context for the LLM.
- **Code Generation:** The LLM receives both the user prompt and retrieved context, improving code relevance and correctness.
- **UI:** Streamlit app manages user interaction, chat state, and code display.

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/NightPrinceY/Cellula-NLP-Engineer-Intern.git
   cd Cellula-NLP-Engineer-Intern/Week3
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download HumanEval dataset:**
   - The required parquet file is included in `data/`.

4. **Set up HuggingFace API key:**
   - Export your HuggingFace API key as an environment variable or update `codegen.py` as needed.

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
- Interact with CodeGenBot in your browser.
- Enter a Python coding prompt (e.g., "Write a function that returns the factorial of a number").
- The bot will retrieve similar problems, generate code, and display the result in a chat interface.

## Folder Structure
```
Week3/
├── app.py                # Streamlit UI
├── pipeline.py           # Main pipeline logic
├── codegen.py            # LLM code generation wrapper
├── embedding.py          # Embedding utility
├── retrieval.py          # Vector search and retrieval
├── data/
│   └── test-00000-of-00001.parquet  # HumanEval dataset
├── requirements.txt      # Dependencies
├── CodeGen.ipynb         # Experiments and prototyping
```

## File Descriptions
- **app.py**: Streamlit app for user interaction and chat UI.
- **pipeline.py**: Orchestrates embedding, retrieval, and code generation.
- **codegen.py**: Handles LLM API calls and prompt construction.
- **embedding.py**: Provides the `Embedder` class for semantic encoding.
- **retrieval.py**: Implements vector search and context retrieval.
- **data/**: Contains the HumanEval dataset in parquet format.
- **CodeGen.ipynb**: Jupyter notebook for prototyping and experiments.
- **requirements.txt**: Python dependencies.

## Example
```python
from pipeline import CodeGenPipeline
pipeline = CodeGenPipeline("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
result = pipeline.generate_code_from_prompt("Write a function that returns the factorial of a number")
print(result)
```

## Next Steps
- Add support for multi-language code generation
- Integrate more datasets and problem types
- Enhance retrieval ranking and context selection
- Deploy as a web service or IDE extension

---
Prepared by: Yahya Alnwsany  
Cellula AI Intern – Week 3  
[Portfolio](https://nightprincey.github.io/Portfolio/) | [Week 3 Repo](https://github.com/NightPrinceY/Cellula-NLP-Engineer-Intern/tree/main/Week3/) 