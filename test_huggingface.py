"""Basic DriverLM test with HuggingFace microsoft/phi-2.

Tests basic prediction, ChainOfThought, and BootstrapFewShot optimization
using DriverLM with a local HuggingFace model.
"""
import dspy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from package.base import DSPyResult, DriverLM

if __name__ == '__main__':
    # Load small HuggingFace model
    model_name = "microsoft/phi-2"  # 2.7B parameters
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device)
    print("Model loaded!")

    # Request function
    def hf_request_fn(prompt: str = None, messages: list[dict] = None, temperature: float = 0.0, max_tokens: int = 256) -> dict:
        if messages:
            # Convert messages to prompt
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

    # Create DriverLM
    lm = DriverLM(
        request_fn=hf_request_fn,
        output_fn=hf_output_fn,
        temperature=0.1,
        max_tokens=100
    )
    dspy.configure(lm=lm)

    # Test basic prediction
    print("\n=== Basic Test ===")
    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2+2?")
    print(f"Q: What is 2+2?\nA: {result.answer}")

    # Test ChainOfThought
    print("\n=== ChainOfThought Test ===")
    cot = dspy.ChainOfThought("sentence -> emotion")
    result = cot(sentence="I love pizza")
    print(f"Sentence: I love pizza")
    print(f"Reasoning: {result.reasoning}")
    print(f"Emotion: {result.emotion}")

    # Test with optimization
    print("\n=== Optimization Test ===")
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

    print("\nTesting optimized model:")
    result = optimized(sentence="I love pizza")
    print(f"Emotion: {result.emotion}")
