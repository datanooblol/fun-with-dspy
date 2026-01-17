"""Load and compare optimized models from all three implementations.

Loads emotion_native_optimized.json, emotion_driver_optimized.json, and
emotion_hf_optimized.json, tests with 10 examples, and compares results
with case-insensitive matching.
"""
import dspy
import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from package.base import DSPyResult, DriverLM

# Setup LMs
ollama_client = httpx.Client(timeout=600.0)

def ollama_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    response = ollama_client.post(
        'http://localhost:11434/api/chat',
        json={"model": "llama3.2-vision:11b", "messages": messages, 
              "stream": False, "options": {"temperature": temperature}}
    )
    response.raise_for_status()
    return {"response": response.json()["message"]["content"]}

def ollama_output_fn(response):
    return DSPyResult(response.get("response", "").strip())

# HuggingFace setup
print("Loading HuggingFace model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True).to(device)

def hf_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = hf_model.generate(**inputs, max_new_tokens=max_tokens, 
                                     temperature=temperature if temperature > 0 else 1.0,
                                     do_sample=temperature > 0, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return {"response": response}

def hf_output_fn(response):
    return DSPyResult(response.get("response", "").strip())

# Create LM instances
lm_native = dspy.LM(model="ollama/llama3.2-vision:11b", api_base="http://localhost:11434", temperature=0.0)
lm_driver = DriverLM(request_fn=ollama_request_fn, output_fn=ollama_output_fn, temperature=0.0)
lm_hf = DriverLM(request_fn=hf_request_fn, output_fn=hf_output_fn, temperature=0.0)

# Test examples
test_examples = [
    "I love pizza!",
    "This is the worst day ever.",
    "The weather is okay.",
    "I'm so excited for the party!",
    "I hate Mondays.",
    "The sky is blue.",
    "This movie is amazing!",
    "I feel terrible about this.",
    "It's just another day.",
    "I'm thrilled to see you!",
]

# Load and test each model
models = [
    ("Native dspy.LM", lm_native, "emotion_native_optimized.json"),
    ("DriverLM", lm_driver, "emotion_driver_optimized.json"),
    ("HuggingFace", lm_hf, "emotion_hf_optimized.json"),
]

results = {}

for name, lm, filepath in models:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    dspy.configure(lm=lm)
    
    # Load optimized model
    class EmotionClassifier(dspy.Module):
        def __init__(self):
            self.predict = dspy.ChainOfThought("sentence -> emotion")
        
        def forward(self, sentence):
            return self.predict(sentence=sentence)
    
    model = EmotionClassifier()
    model.load(filepath)
    
    # Test
    results[name] = []
    for sentence in test_examples:
        result = model(sentence=sentence)
        emotion = result.emotion.lower()  # Normalize to lowercase
        results[name].append(emotion)
        print(f"'{sentence}' -> {emotion}")

# Comparison table
print(f"\n{'='*60}")
print("COMPARISON TABLE")
print('='*60)
print(f"{'Sentence':<40} {'Native':<10} {'Driver':<10} {'HF':<10}")
print('-'*60)
for i, sentence in enumerate(test_examples):
    native = results["Native dspy.LM"][i]
    driver = results["DriverLM"][i]
    hf = results["HuggingFace"][i]
    print(f"{sentence:<40} {native:<10} {driver:<10} {hf:<10}")

# Agreement stats (case-insensitive)
print(f"\n{'='*60}")
print("AGREEMENT STATISTICS (case-insensitive)")
print('='*60)
native_driver_match = sum(1 for i in range(len(test_examples)) 
                          if results["Native dspy.LM"][i].lower() == results["DriverLM"][i].lower())
print(f"Native vs Driver: {native_driver_match}/{len(test_examples)} match ({native_driver_match/len(test_examples)*100:.0f}%)")

native_hf_match = sum(1 for i in range(len(test_examples)) 
                      if results["Native dspy.LM"][i].lower() == results["HuggingFace"][i].lower())
print(f"Native vs HF: {native_hf_match}/{len(test_examples)} match ({native_hf_match/len(test_examples)*100:.0f}%)")

driver_hf_match = sum(1 for i in range(len(test_examples)) 
                      if results["DriverLM"][i].lower() == results["HuggingFace"][i].lower())
print(f"Driver vs HF: {driver_hf_match}/{len(test_examples)} match ({driver_hf_match/len(test_examples)*100:.0f}%)")
