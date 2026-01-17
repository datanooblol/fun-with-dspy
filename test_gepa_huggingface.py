"""GEPA optimization using DriverLM with HuggingFace microsoft/phi-2.

Tests GEPA with a local HuggingFace model instead of Ollama.
Uses same training data and structure as Ollama tests for comparison.
Saves to emotion_hf_optimized.json.
"""
import dspy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dspy.teleprompt import GEPA
from package.base import DSPyResult, DriverLM

# Load HuggingFace model once
print("Loading microsoft/phi-2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True
).to(device)
print(f"Model loaded on {device}\n")

def hf_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    # Convert messages to prompt
    prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return {"response": response}

def hf_output_fn(response):
    return DSPyResult(response.get("response", "").strip())

# Create DriverLM instances
lm_hf = DriverLM(
    request_fn=hf_request_fn,
    output_fn=hf_output_fn,
    temperature=0.0
)

reflection_hf = DriverLM(
    request_fn=hf_request_fn,
    output_fn=hf_output_fn,
    temperature=1.0
)

print("="*60)
print("Testing GEPA with HuggingFace microsoft/phi-2")
print("="*60)

dspy.configure(lm=lm_hf)

# Training data (same as Ollama test)
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
print("\nStarting GEPA optimization...")
optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=1,  # HF models don't parallelize well
    track_stats=True,
    reflection_lm=reflection_hf
)

optimized = optimizer.compile(
    EmotionClassifier(),
    trainset=train,
    valset=val
)

# Save
optimized.save("emotion_hf_optimized.json")
print("\nSaved to emotion_hf_optimized.json")

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

print("\n" + "="*60)
print("GEPA OPTIMIZATION COMPLETE")
print("="*60)
