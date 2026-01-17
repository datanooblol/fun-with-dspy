"""Comprehensive comparison of three LM implementations.

Tests native dspy.LM (Ollama), DriverLM (Ollama), and DriverLM (HuggingFace phi-2)
with basic prediction, ChainOfThought, and BootstrapFewShot optimization.
"""
import dspy
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from package.base import DSPyResult, DriverLM

def test_model(lm, name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    dspy.configure(lm=lm)
    
    # Test 1: Basic prediction
    print("\n--- Basic Prediction ---")
    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2+2?")
    print(f"Q: What is 2+2?\nA: {result.answer}")
    
    # Test 2: ChainOfThought
    print("\n--- ChainOfThought ---")
    cot = dspy.ChainOfThought("sentence -> emotion")
    result = cot(sentence="I love pizza")
    print(f"Sentence: I love pizza")
    print(f"Emotion: {result.emotion}")
    
    # Test 3: Optimization
    print("\n--- After Optimization ---")
    train = [
        dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
        dspy.Example(sentence="This is terrible.", emotion="sad").with_inputs("sentence"),
    ]
    
    class EmotionClassifier(dspy.Module):
        def __init__(self):
            self.predict = dspy.ChainOfThought("sentence -> emotion")
        def forward(self, sentence):
            return self.predict(sentence=sentence)
    
    def metric(example, pred, trace=None):
        return example.emotion.lower() == pred.emotion.lower()
    
    from dspy.teleprompt import BootstrapFewShot
    optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
    optimized = optimizer.compile(EmotionClassifier(), trainset=train)
    
    test_cases = ["I love pizza", "I hate Mondays", "The sky is blue"]
    for test in test_cases:
        result = optimized(sentence=test)
        print(f"'{test}' -> {result.emotion}")

if __name__ == '__main__':
    # 1. Native dspy.LM with Ollama
    print("\n" + "="*60)
    print("SETUP: Native dspy.LM with Ollama")
    print("="*60)
    lm_native = dspy.LM(
        model="ollama/llama3.2-vision:11b",
        api_base="http://localhost:11434",
        temperature=0.0
    )
    test_model(lm_native, "Native dspy.LM (Ollama llama3.2-vision:11b)")
    
    # 2. DriverLM with Ollama
    print("\n" + "="*60)
    print("SETUP: DriverLM with Ollama")
    print("="*60)
    
    def ollama_request_fn(prompt: str = None, messages: list[dict] = None, temperature: float = 0.0, max_tokens: int = 256) -> dict:
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
    
    def ollama_output_fn(response: dict) -> DSPyResult:
        return DSPyResult(response.get("response", "").strip())
    
    lm_driver = DriverLM(
        request_fn=ollama_request_fn,
        output_fn=ollama_output_fn,
        temperature=0.0
    )
    test_model(lm_driver, "DriverLM (Ollama llama3.2-vision:11b)")
    
    # 3. DriverLM with HuggingFace
    print("\n" + "="*60)
    print("SETUP: DriverLM with HuggingFace")
    print("="*60)
    
    model_name = "microsoft/phi-2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    print("Model loaded!")
    
    def hf_request_fn(prompt: str = None, messages: list[dict] = None, temperature: float = 0.0, max_tokens: int = 256) -> dict:
        if messages:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return {"response": response.strip()}
    
    def hf_output_fn(response: dict) -> DSPyResult:
        return DSPyResult(response.get("response", "").strip())
    
    lm_hf = DriverLM(
        request_fn=hf_request_fn,
        output_fn=hf_output_fn,
        temperature=0.1,
        max_tokens=100
    )
    test_model(lm_hf, "DriverLM (HuggingFace microsoft/phi-2)")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
