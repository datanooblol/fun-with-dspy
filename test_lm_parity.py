"""Simple parity test between native dspy.LM and DriverLM.

Verifies both produce identical outputs for the same input with system messages.
"""
import dspy
import httpx
from package.base import DSPyResult, DriverLM

# Create httpx client
ollama_client = httpx.Client(timeout=600.0)

# Setup native dspy.LM
lm_native = dspy.LM(
    model="ollama/llama3.2-vision:11b",
    api_base="http://localhost:11434",
    temperature=0.0
)

# Setup DriverLM
def ollama_request_fn(prompt: str = None, messages: list[dict] = None, temperature: float = 0.0, max_tokens: int = 256) -> dict:
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    response = ollama_client.post(
        'http://localhost:11434/api/chat',
        json={
            "model": "llama3.2-vision:11b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    response.raise_for_status()
    return {"response": response.json()["message"]["content"]}

def ollama_output_fn(response: dict) -> DSPyResult:
    return DSPyResult(response.get("response", "").strip())

lm_driver = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=ollama_output_fn,
    temperature=0.0
)

# Test with same messages
test_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2? Answer with just the number."}
]

print("Testing with identical messages...")
print("\n1. Native dspy.LM:")
result_native = lm_native(messages=test_messages)
print(f"Result: {result_native}")

print("\n2. DriverLM:")
result_driver = lm_driver(messages=test_messages)
print(f"Result: {result_driver}")

print("\n3. Comparing outputs:")
print(f"Native output: '{result_native[0]}'")
print(f"Driver output: '{result_driver[0]}'")
print(f"Match: {result_native[0] == result_driver[0]}")
