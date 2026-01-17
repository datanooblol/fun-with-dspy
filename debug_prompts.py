import dspy
import requests
from package.base import DSPyResult, DriverLM

# Debug request function
def debug_request_fn(prompt: str = None, messages: list[dict] = None, temperature: float = 0.0, max_tokens: int | None = None) -> dict:
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    print("\n=== Messages sent to Ollama ===")
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:200]}...")
    
    response = requests.post(
        'http://localhost:11434/api/chat',
        json={
            "model": "llama3.2-vision:11b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    response.raise_for_status()
    result = response.json()["message"]["content"]
    print(f"\nResponse: {result[:200]}...")
    return {"response": result}

def llama_output_fn(response: dict) -> DSPyResult:
    return DSPyResult(response.get("response", "").strip())

# Test
print("="*60)
print("Testing Native dspy.LM")
print("="*60)
lm_native = dspy.LM(
    model="ollama/llama3.2-vision:11b",
    api_base="http://localhost:11434",
    temperature=0.0
)
dspy.configure(lm=lm_native)
classifier = dspy.ChainOfThought("sentence -> emotion")
result = classifier(sentence="I love pizza")
print(f"\nFinal: {result.emotion}")

print("\n" + "="*60)
print("Testing Custom DriverLM")
print("="*60)
lm_driver = DriverLM(
    request_fn=debug_request_fn,
    output_fn=llama_output_fn,
    temperature=0.0
)
dspy.configure(lm=lm_driver)
classifier = dspy.ChainOfThought("sentence -> emotion")
result = classifier(sentence="I love pizza")
print(f"\nFinal: {result.emotion}")
