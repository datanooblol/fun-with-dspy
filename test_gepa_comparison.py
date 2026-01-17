"""GEPA comparison between native dspy.LM and DriverLM.

Runs GEPA optimization with both implementations using the same Ollama model.
Uses httpx for connection pooling. Clears cache between tests for fair comparison.
Saves to emotion_native_optimized.json and emotion_driver_optimized.json.
"""
import dspy
import httpx
from dspy.teleprompt import GEPA
from package.base import DSPyResult, DriverLM

# Create a global httpx client for connection pooling (same as LiteLLM)
ollama_client = httpx.Client(timeout=600.0)

def run_gepa_test(lm, reflection_lm, name, save_file):
    print(f"\n{'='*60}")
    print(f"Testing GEPA with {name}")
    print('='*60)
    
    dspy.configure(lm=lm)
    
    # Training data
    train = [
        dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
        dspy.Example(sentence="This is terrible.", emotion="sad").with_inputs("sentence"),
        dspy.Example(sentence="The weather is okay.", emotion="neutral").with_inputs("sentence"),
    ]
    
    val = [
        dspy.Example(sentence="I'm so excited!", emotion="happy").with_inputs("sentence"),
        dspy.Example(sentence="I hate Mondays.", emotion="sad").with_inputs("sentence"),
        dspy.Example(sentence="The sky is blue.", emotion="neutral").with_inputs("sentence"),
    ]
    
    # Module
    class EmotionClassifier(dspy.Module):
        def __init__(self):
            self.predict = dspy.ChainOfThought("sentence -> emotion")
        
        def forward(self, sentence):
            return self.predict(sentence=sentence)
    
    # Metric with feedback
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        correct = example.emotion.lower() == pred.emotion.lower()
        score = 1.0 if correct else 0.0
        
        if pred_name is None:
            return score
        
        if correct:
            feedback = f"Correct! You classified '{example.sentence}' as '{pred.emotion}', which matches the gold label '{example.emotion}'."
        else:
            feedback = f"Incorrect. You classified '{example.sentence}' as '{pred.emotion}', but the correct emotion is '{example.emotion}'. Think about the emotional tone more carefully."
        
        return dspy.Prediction(score=score, feedback=feedback)
    
    # Optimize
    print("Starting GEPA optimization...")
    optimizer = GEPA(
        metric=metric_with_feedback,
        auto="light",
        num_threads=4,
        track_stats=True,
        reflection_lm=reflection_lm
    )
    
    optimized = optimizer.compile(
        EmotionClassifier(),
        trainset=train,
        valset=val
    )
    
    # Save
    optimized.save(save_file)
    print(f"\nSaved to {save_file}")
    
    # Test
    print("\n=== Testing Optimized Model ===")
    test_cases = ["I love pizza", "I hate Mondays", "The sky is blue"]
    for test in test_cases:
        result = optimized(sentence=test)
        print(f"'{test}' -> {result.emotion}")
    
    # Show prompt preview
    print("\n=== Optimized Prompt Preview ===")
    for pred_name, pred in optimized.named_predictors():
        print(f"\n{pred_name}:")
        print(pred.signature.instructions[:300] + "...")

if __name__ == '__main__':
    # Clear cache before starting
    import dspy
    if hasattr(dspy, 'cache'):
        dspy.cache.reset_memory_cache()
        print("Cache cleared before tests\n")
    
    # Setup request/output functions (shared by both tests)
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
    
    # 1. Native dspy.LM (both student and reflection)
    lm_native = dspy.LM(
        model="ollama/llama3.2-vision:11b",
        api_base="http://localhost:11434",
        temperature=0.0
    )
    reflection_native = dspy.LM(
        model="ollama/llama3.2-vision:11b",
        api_base="http://localhost:11434",
        temperature=1.0
    )
    run_gepa_test(lm_native, reflection_native, "Native dspy.LM", "emotion_native_optimized.json")
    
    # Clear cache between tests
    if hasattr(dspy, 'cache'):
        dspy.cache.reset_memory_cache()
        print("\nCache cleared between tests\n")
    
    # 2. Custom DriverLM (both student and reflection)
    lm_driver = DriverLM(
        request_fn=ollama_request_fn,
        output_fn=ollama_output_fn,
        temperature=0.0
    )
    reflection_driver = DriverLM(
        request_fn=ollama_request_fn,
        output_fn=ollama_output_fn,
        temperature=1.0
    )
    run_gepa_test(lm_driver, reflection_driver, "Custom DriverLM", "emotion_driver_optimized.json")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
