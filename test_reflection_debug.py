"""Debug reflection LM behavior at temperature=1.0.

Tests both native dspy.LM and DriverLM with temperature=1.0 (reflection mode)
to verify stochastic behavior and cache functionality.
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

# Test with same messages at temperature=1.0 (reflection temperature)
test_messages = [
    {"role": "user", "content": "Classify the emotion: I love this!"}
]

print("Testing at temperature=1.0 (reflection mode):\n")

# Native
lm_native = dspy.LM(
    model="ollama/llama3.2-vision:11b",
    api_base="http://localhost:11434",
    temperature=1.0,
    cache=False  # Disable cache for testing
)

# Driver
lm_driver = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=ollama_output_fn,
    temperature=1.0,
    cache=False  # Disable cache for testing
)

print("1. Native dspy.LM (5 calls):")
for i in range(5):
    result = lm_native(messages=test_messages)
    print(f"  Call {i+1}: {result[0][:100]}")

print("\n2. DriverLM (5 calls):")
for i in range(5):
    result = lm_driver(messages=test_messages)
    print(f"  Call {i+1}: {result[0][:100]}")

print("\n3. Testing cache behavior:")
lm_native_cached = dspy.LM(
    model="ollama/llama3.2-vision:11b",
    api_base="http://localhost:11434",
    temperature=1.0,
    cache=True
)
lm_driver_cached = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=ollama_output_fn,
    temperature=1.0,
    cache=True
)

print("\nNative with cache (should be same):")
r1 = lm_native_cached(messages=test_messages)
r2 = lm_native_cached(messages=test_messages)
print(f"  First:  {r1[0][:50]}")
print(f"  Second: {r2[0][:50]}")
print(f"  Match: {r1[0] == r2[0]}")

print("\nDriver with cache (should be same):")
r1 = lm_driver_cached(messages=test_messages)
r2 = lm_driver_cached(messages=test_messages)
print(f"  First:  {r1[0][:50]}")
print(f"  Second: {r2[0][:50]}")
print(f"  Match: {r1[0] == r2[0]}")
