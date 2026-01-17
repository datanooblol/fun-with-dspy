"""Test comparison between native dspy.LM and DriverLM with Ollama.

Compares basic emotion classification before and after BootstrapFewShot optimization.
Uses requests library (no connection pooling).
"""
import dspy
import requests
from package.base import DSPyResult, DriverLM

# Request and output functions
def llama_request_fn(prompt: str = None, messages: list[dict] = None, temperature: float = 0.0, max_tokens: int | None = None) -> dict:
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
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
    return {"response": response.json()["message"]["content"]}

def llama_output_fn(response: dict) -> DSPyResult:
    return DSPyResult(response.get("response", "").strip())

# Training data
train = [
    dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
    dspy.Example(sentence="This is terrible.", emotion="sad").with_inputs("sentence"),
    dspy.Example(sentence="The weather is okay.", emotion="neutral").with_inputs("sentence"),
]

# Test cases
test_cases = [
    "I love pizza",
    "I hate Mondays",
    "The sky is blue"
]

# Metric
def emotion_metric(example, pred, trace=None):
    return example.emotion.lower() == pred.emotion.lower()

# Test function
def test_lm(lm, name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    dspy.configure(lm=lm)
    
    # Define module
    class EmotionClassifier(dspy.Module):
        def __init__(self):
            self.predict = dspy.ChainOfThought("sentence -> emotion")
        
        def forward(self, sentence):
            return self.predict(sentence=sentence)
    
    # Before optimization
    print("\n--- Before Optimization ---")
    classifier = EmotionClassifier()
    for test in test_cases:
        result = classifier(sentence=test)
        print(f"'{test}' -> {result.emotion}")
    
    # Optimize
    print("\n--- Optimizing... ---")
    from dspy.teleprompt import BootstrapFewShot
    optimizer = BootstrapFewShot(metric=emotion_metric, max_bootstrapped_demos=2)
    optimized = optimizer.compile(EmotionClassifier(), trainset=train)
    
    # After optimization
    print("\n--- After Optimization ---")
    for test in test_cases:
        result = optimized(sentence=test)
        print(f"'{test}' -> {result.emotion}")

if __name__ == "__main__":
    # Run tests
    lm_native = dspy.LM(
        model="ollama/llama3.2-vision:11b",
        api_base="http://localhost:11434",
        temperature=0.0
    )

    lm_driver = DriverLM(
        request_fn=llama_request_fn,
        output_fn=llama_output_fn,
        temperature=0.0
    )

    test_lm(lm_native, "Native dspy.LM")
    test_lm(lm_driver, "Custom DriverLM")
