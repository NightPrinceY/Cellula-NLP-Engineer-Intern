import requests

def llama_guard_filter(text):
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-guard-3-8b",
        "max_tokens": 25,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict safety classifier. Respond with only one word: either 'safe' or 'unsafe'. "
                    "Do not add punctuation, explanation, or formatting."
                )
            },
            {"role": "user", "content": text}
        ]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=10)
        content = response.json()["choices"][0]["message"]["content"].strip().lower()
        if "safe" in content:
            return "safe"
        else:
            return "unsafe"
    except Exception as e:
        print("Llama Guard API Error:", e)
        return "api_error"
