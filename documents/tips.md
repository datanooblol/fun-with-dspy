# DSPy Best Practices & Tips

## 1. Saving Optimized Models

**Always save your optimized models** after running optimizers like GEPA, BootstrapFewShot, etc.

```python
from dspy.teleprompt import GEPA

optimizer = GEPA(metric=metric, auto="light", reflection_lm=reflection_lm)
optimized = optimizer.compile(student, trainset=train, valset=val)

# Save immediately after optimization
optimized.save("model_optimized.json")
```

**Why?** Optimization can take minutes to hours. Saving ensures you don't lose your work if something crashes.

## 2. Loading Optimized Models

**You must recreate the module class** before loading:

```python
import dspy

# Define the SAME class structure used during optimization
class EmotionClassifier(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought("sentence -> emotion")
    
    def forward(self, sentence):
        return self.predict(sentence=sentence)

# Create instance and load
model = EmotionClassifier()
model.load("model_optimized.json")

# Now you can use it
result = model(sentence="I love this!")
```

**Important**: The class definition must match the structure used during optimization. DSPy saves the optimized prompts/instructions, not the code.

## 3. Comparing Results: Preprocess Before Evaluation

**Never compare raw outputs directly.** Always preprocess to normalize differences:

### Case Normalization

```python
# Bad - case-sensitive comparison
if pred.emotion == gold.emotion:  # "Happy" != "happy"
    score = 1.0

# Good - case-insensitive comparison
if pred.emotion.lower() == gold.emotion.lower():  # "Happy" == "happy"
    score = 1.0
```

### Common Preprocessing Steps

```python
def preprocess(text):
    """Normalize text for fair comparison"""
    text = text.lower()                    # Lowercase
    text = text.strip()                    # Remove whitespace
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    return text

# Compare preprocessed outputs
if preprocess(pred.emotion) == preprocess(gold.emotion):
    score = 1.0
```

### Advanced Preprocessing

For more complex comparisons:

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

def advanced_preprocess(text):
    """Stemming + tokenization"""
    text = text.lower()
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed

# Example: "running" and "runs" both become "run"
pred_tokens = advanced_preprocess(pred.text)
gold_tokens = advanced_preprocess(gold.text)
```

## 4. Choosing Evaluation Metrics

Different tasks need different metrics:

### Exact Match (Classification)

```python
def metric(example, pred):
    return 1.0 if pred.emotion.lower() == example.emotion.lower() else 0.0
```

### ROUGE Score (Text Generation)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def metric(example, pred):
    scores = scorer.score(example.answer, pred.answer)
    return scores['rougeL'].fmeasure
```

### Accuracy, Precision, Recall (Multi-class)

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, testset):
    predictions = [model(ex.input).label.lower() for ex in testset]
    gold_labels = [ex.label.lower() for ex in testset]
    
    accuracy = accuracy_score(gold_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### Semantic Similarity (Embeddings)

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def metric(example, pred):
    emb1 = model.encode(example.answer, convert_to_tensor=True)
    emb2 = model.encode(pred.answer, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return similarity
```

## 5. Real-World Example: Complete Workflow

```python
import dspy
from dspy.teleprompt import GEPA

# 1. Define module
class EmotionClassifier(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought("sentence -> emotion")
    
    def forward(self, sentence):
        return self.predict(sentence=sentence)

# 2. Optimize
optimizer = GEPA(metric=metric_with_feedback, auto="light", reflection_lm=reflection_lm)
optimized = optimizer.compile(EmotionClassifier(), trainset=train, valset=val)

# 3. Save
optimized.save("emotion_optimized.json")

# 4. Load (later, in production)
model = EmotionClassifier()
model.load("emotion_optimized.json")

# 5. Evaluate with preprocessing
def evaluate(model, testset):
    correct = 0
    for example in testset:
        pred = model(sentence=example.sentence)
        # Preprocess before comparison
        if pred.emotion.lower().strip() == example.emotion.lower().strip():
            correct += 1
    return correct / len(testset)

accuracy = evaluate(model, test_data)
print(f"Accuracy: {accuracy:.2%}")
```

## Key Takeaways

1. **Always save** after optimization - it's expensive to rerun
2. **Recreate the class** before loading - DSPy needs the structure
3. **Preprocess outputs** before comparison - normalize case, whitespace, punctuation
4. **Choose appropriate metrics** - exact match for classification, ROUGE for generation, embeddings for semantic similarity
5. **Be consistent** - use the same preprocessing in training metrics and evaluation

## Common Pitfalls

❌ **Don't**: Compare raw outputs without preprocessing
```python
# This fails on "Happy" vs "happy"
if pred.emotion == gold.emotion:
    score = 1.0
```

✅ **Do**: Normalize before comparison
```python
# This handles case differences
if pred.emotion.lower() == gold.emotion.lower():
    score = 1.0
```

❌ **Don't**: Forget to save optimized models
```python
optimized = optimizer.compile(student, trainset=train)
# Program crashes, optimization lost!
```

✅ **Do**: Save immediately
```python
optimized = optimizer.compile(student, trainset=train)
optimized.save("model.json")  # Safe!
```

❌ **Don't**: Try to load without the class
```python
# This won't work
model = dspy.load("model.json")
```

✅ **Do**: Create instance first
```python
model = EmotionClassifier()
model.load("model.json")
```
