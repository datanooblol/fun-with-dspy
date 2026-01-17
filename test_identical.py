"""Verify dspy.LM and DriverLM produce identical outputs.

Tests that both implementations return the same results for identical inputs
and that caching works correctly for both.
"""
import dspy
import httpx
from package.base import DSPyResult, DriverLM

ollama_client = httpx.Client(timeout=600.0)

def ollama_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
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

def ollama_output_fn(response):
    return DSPyResult(response.get("response", "").strip())

# Clear cache
if hasattr(dspy, 'cache'):
    dspy.cache.reset_memory_cache()

# Test identical calls
test_messages = [{"role": "user", "content": "What is 2+2?"}]

lm_native = dspy.LM(
    model="ollama/llama3.2-vision:11b",
    api_base="http://localhost:11434",
    temperature=0.0
)

lm_driver = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=ollama_output_fn,
    temperature=0.0
)

print("Test 1: Same input, same output?")
r1 = lm_native(messages=test_messages)
r2 = lm_driver(messages=test_messages)
print(f"Native: {r1[0]}")
print(f"Driver: {r2[0]}")
print(f"Match: {r1[0] == r2[0]}\n")

print("Test 2: Cache working?")
r3 = lm_native(messages=test_messages)
r4 = lm_driver(messages=test_messages)
print(f"Native cached: {r3[0] == r1[0]}")
print(f"Driver cached: {r4[0] == r2[0]}")
