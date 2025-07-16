from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

load_dotenv()
API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def generate_code_with_context(user_prompt, context=None):
    client = InferenceClient(provider="nscale", api_key=API_KEY)
    if context:
        final_prompt = f"{context}\n\n# Your Task:\n{user_prompt}\n"
    else:
        final_prompt = user_prompt
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": final_prompt}],
    )
    return completion.choices[0].message.content 