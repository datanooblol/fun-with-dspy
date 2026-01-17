"""Compare evaluation scores between native dspy.LM and DriverLM.

Tests both implementations on simple emotion classification task and
compares their evaluation scores using dspy.Evaluate.
"""
import dspy
import httpx
from package.base import DSPyResult, DriverLM

ollama_client = httpx.Client(timeout=600.0)

def test_with_lm(lm, name):
    print(f"\n{'='*60}")
    print(f"Testing with {name}")
    print('='*60)
    
    dspy.configure(lm=lm)
    
    # Simple test data
    train = [
        dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
        dspy.Example(sentence="This is terrible.", emotion="sad").with_inputs("sentence"),
    ]
    
    class EmotionClassifier(dspy.Module):
        def __init__(self):
            self.predict = dspy.ChainOfThought("sentence -> emotion")
        
        def forward(self, sentence):
            return self.predict(sentence=sentence)
    
    # Test before optimization
    model = EmotionClassifier()
    print("\nBefore optimization:")
    for ex in train:
        result = model(sentence=ex.sentence)
        print(f"  '{ex.sentence}' -> {result.emotion} (expected: {ex.emotion})")
    
    # Simple metric
    def metric(example, pred, trace=None):
        return 1.0 if example.emotion.lower() == pred.emotion.lower() else 0.0
    
    # Evaluate
    from dspy.evaluate import Evaluate
    evaluator = Evaluate(devset=train, metric=metric, num_threads=1)
    score = evaluator(model)
    print(f"\nScore: {score}")
    
    return score

# Test both
lm_native = dspy.LM(
    model="ollama/llama3.2-vision:11b",
    api_base="http://localhost:11434",
    temperature=0.0
)

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

lm_driver = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=ollama_output_fn,
    temperature=0.0
)

score_native = test_with_lm(lm_native, "Native dspy.LM")
score_driver = test_with_lm(lm_driver, "DriverLM")

print(f"\n{'='*60}")
print(f"Native score: {score_native}")
print(f"Driver score: {score_driver}")
print(f"Match: {score_native.score == score_driver.score}")
print('='*60)
