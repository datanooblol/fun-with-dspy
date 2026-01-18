# DSPy Optimization Guide

## Overview

DSPy optimization automatically improves prompts by finding better instructions and/or demonstrations (few-shot examples). This document covers both pre-built optimizers and custom optimization strategies.

---

## Core Concept: What Gets Optimized?

All DSPy optimizers ultimately modify **predictors** within your module by setting:

1. **`predictor.demos`** - Few-shot examples
2. **`predictor.signature.instructions`** - System instructions

**Key insight:** `optimizer.compile()` returns a NEW module with optimized predictors, leaving the original unchanged.

---

## Accessing Predictors

### Using named_predictors()

```python
class EmotionClassifier(dspy.Module):
    def __init__(self):
        self.program = dspy.ChainOfThought(Emotion)
    
    def forward(self, sentence):
        return self.program(sentence=sentence)

# After optimization
optimized = optimizer.compile(EmotionClassifier(), trainset=trainset)

# Inspect what was optimized
for name, predictor in optimized.named_predictors():
    print(f"{name}:")
    print(f"  Signature: {predictor.signature}")
    print(f"  Demos: {len(predictor.demos)} examples")
```

**Output:**
```
program.predict:
  Signature: StringSignature(sentence -> reasoning, emotion)
  Demos: 2 examples
```

### Direct Access

```python
# For dspy.Predict
predictor = dspy.Predict(Signature)
predictor.demos = [demo1, demo2]

# For dspy.ChainOfThought (wraps Predict internally)
cot = dspy.ChainOfThought(Signature)
cot.predict.demos = [demo1, demo2]  # Access internal predictor
```

---

## Pre-Built Optimizers

### 1. BootstrapFewShot

**Purpose:** Find best few-shot examples by running teacher model and validating outputs

**How it works:**
1. Run teacher model on each training example
2. Validate output using metric function
3. Keep successful (input, output) pairs as demos
4. Assign demos to student predictors

**Code:**
```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    max_bootstrapped_demos=2,  # Max demos to generate
    max_labeled_demos=2         # Max labeled examples to use
)

optimized = optimizer.compile(
    EmotionClassifier(),
    trainset=trainset
)
```

**What happens during compilation:**
```
Iteration 1: Run teacher on trainset[0]
  Input: "I love this!"
  Output: emotion="happy", reasoning="..."
  Metric: PASS ✓
  → Add to demo pool

Iteration 2: Run teacher on trainset[1]
  Input: "This is terrible."
  Output: emotion="sad", reasoning="..."
  Metric: PASS ✓
  → Add to demo pool

Result: optimized.program.predict.demos = [demo1, demo2]
```

**Demo structure:**
```python
Example({
    'augmented': True,
    'sentence': 'I love this!',
    'reasoning': 'The use of exclamation mark and word "love" convey positive sentiment.',
    'emotion': 'happy'
})
```

**Key parameters:**
- `max_bootstrapped_demos`: How many demos to generate from trainset
- `max_labeled_demos`: How many pre-labeled examples to include
- `metric`: Function to validate outputs (optional)
- `teacher_settings`: Dict of LM settings for teacher

---

### 2. MIPRO (Multi-prompt Instruction Proposal Optimizer)

**Purpose:** Optimize both instructions AND demos simultaneously

**How it works:**
1. Generate instruction candidates using LLM
2. Sample demo combinations
3. Evaluate all (instruction, demos) pairs
4. Select best combination based on metric

**Code:**
```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=my_metric,
    num_candidates=10,      # Instruction candidates to try
    init_temperature=1.0    # LLM temperature for generation
)

optimized = optimizer.compile(
    EmotionClassifier(),
    trainset=trainset,
    num_trials=20           # Total combinations to try
)
```

**What gets optimized:**
- `predictor.signature.instructions` - New instruction text
- `predictor.demos` - Best demo combination

---

### 3. GEPA (Generalized Evolutionary Prompt Adaptation)

**Purpose:** Evolutionary optimization of instructions with reflection

**How it works:**
1. Start with base program
2. Use reflection LM to propose instruction improvements
3. Evaluate on validation set
4. Keep improvements, discard failures
5. Iterate until convergence or budget exhausted

**Code:**
```python
from dspy.teleprompt import GEPA

# Metric with feedback for reflection
def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    correct = example.emotion.lower() == pred.emotion.lower()
    score = 1.0 if correct else 0.0
    
    if pred_name is None:
        return score
    
    if correct:
        feedback = f"Correct! You classified '{example.sentence}' as '{pred.emotion}'."
    else:
        feedback = f"Incorrect. You classified '{example.sentence}' as '{pred.emotion}', but correct is '{example.emotion}'."
    
    return dspy.Prediction(score=score, feedback=feedback)

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",              # Optimization intensity
    num_threads=4,
    track_stats=True,
    reflection_lm=reflection_lm  # Separate LM for generating improvements
)

optimized = optimizer.compile(
    EmotionClassifier(),
    trainset=train,
    valset=val
)
```

**Optimization process:**
```
Iteration 0: Base program score: 0.33 (1/3 correct)

Iteration 1: Reflection LM proposes new instruction
  "Consider tone markers like exclamation marks..."
  New score: 0.67 (2/3 correct) ✓ KEEP

Iteration 3: Reflection LM proposes refinement
  "Analyze sentence structure, punctuation, emotional cues..."
  New score: 1.0 (3/3 correct) ✓ KEEP

Iterations 4-125: Score remains 1.0, no improvements found
```

**Result:**
```python
optimized.program.predict.signature.instructions = """
# Instruction for the assistant

## Task Description
Given a sentence, classify the sentiment into: 'happy', 'sad', 'neutral', or 'frustration'.

## Task Requirements
1. Identify the sentiment
2. Classify into one of four emotions
3. Provide reasoning with analysis of structure, punctuation, emotional cues
4. Consider emotional adverbs and adjectives
5. Tone markers (!, ?) influence emotional tone

## Niche Information
1. Strong negative opinions → frustration/sadness
2. Positive opinions → happiness
3. Neutral sentences lack strong connotations
...
"""
```

---

## Custom Optimization: Grid Search

### Concept

Manually test different combinations of:
- Number of demos (k)
- Demo selection strategies
- Temperature settings
- Instruction variations

### Implementation

```python
def grid_search_optimization(
    module_class,
    trainset,
    valset,
    metric,
    demo_counts=[0, 1, 2, 3],
    temperatures=[0.0, 0.3, 0.7]
):
    """
    Grid search over demos and temperature.
    """
    best_score = 0
    best_config = None
    results = []
    
    for k in demo_counts:
        for temp in temperatures:
            # Create fresh module
            module = module_class()
            
            # Set demos
            if k > 0:
                demos = trainset[:k]
                for name, predictor in module.named_predictors():
                    predictor.demos = demos
            
            # Configure LM with temperature
            lm = dspy.LM(model="...", temperature=temp)
            dspy.configure(lm=lm)
            
            # Evaluate
            score = evaluate_module(module, valset, metric)
            
            results.append({
                'k': k,
                'temp': temp,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_config = {'k': k, 'temp': temp}
                best_module = module
    
    return best_module, best_config, results
```

### Demo Selection Strategies

**Random sampling:**
```python
import random
demos = random.sample(trainset, k=3)
```

**Diversity-based:**
```python
# Select demos covering different classes
demos = []
for emotion in ['happy', 'sad', 'neutral']:
    examples = [ex for ex in trainset if ex.emotion == emotion]
    demos.append(random.choice(examples))
```

**Similarity-based (KNN-style):**
```python
from sklearn.metrics.pairwise import cosine_similarity

def select_similar_demos(query, trainset, k=3):
    # Embed query and trainset
    query_emb = embed(query)
    train_embs = [embed(ex.sentence) for ex in trainset]
    
    # Find k most similar
    similarities = cosine_similarity([query_emb], train_embs)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]
    
    return [trainset[i] for i in top_k_idx]
```

---

## How Demos Are Constructed

### Creating Demos

```python
# Method 1: From labeled data
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
).with_inputs("sentence")  # Mark which fields are inputs

# Method 2: From model outputs (bootstrapping)
prediction = teacher(sentence="I love this!")
demo = dspy.Example(
    sentence="I love this!",
    emotion=prediction.emotion,
    reasoning=prediction.reasoning
).with_inputs("sentence")
```

### Demo Format in Prompts

**Without demos:**
```
System: Your input fields are: sentence
        Your output fields are: emotion
        ...

User: [[ ## sentence ## ]]
      I love pizza
```

**With demos:**
```
System: Your input fields are: sentence
        Your output fields are: emotion
        ...

User: [[ ## sentence ## ]]
      I love this!

Assistant: [[ ## emotion ## ]]
           happy
           [[ ## completed ## ]]

User: [[ ## sentence ## ]]
      This is terrible.

Assistant: [[ ## emotion ## ]]
           sad
           [[ ## completed ## ]]

User: [[ ## sentence ## ]]
      I love pizza
```

### Demo Construction in Each Round

**BootstrapFewShot process:**

```
Round 1:
  Input: trainset[0] = "I love this!"
  Teacher output: emotion="happy", reasoning="..."
  Metric validation: PASS
  → Create demo: Example(sentence="I love this!", emotion="happy", reasoning="...")
  → Add to demo pool

Round 2:
  Input: trainset[1] = "This is terrible."
  Teacher output: emotion="sad", reasoning="..."
  Metric validation: PASS
  → Create demo: Example(sentence="This is terrible.", emotion="sad", reasoning="...")
  → Add to demo pool

Round 3:
  Input: trainset[2] = "It's okay."
  Teacher output: emotion="happy", reasoning="..."
  Metric validation: FAIL (expected "neutral")
  → Discard, don't add to demos

Final: predictor.demos = [demo_from_round1, demo_from_round2]
```

**Key points:**
- Each successful round adds ONE demo
- Demos include both inputs AND outputs from teacher
- Failed validations are discarded
- Demos are stored with `augmented=True` flag

---

## Saving and Loading Optimized Modules

### Save

```python
from pathlib import Path

artifact_dir = Path("./artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)

# Save optimized module
filename = artifact_dir / "optimized_classifier.json"
optimized.save(filename)
```

### Load

```python
# Create fresh module instance
loaded = EmotionClassifier()

# Load optimized state
loaded.load(filename)

# Use it
result = loaded(sentence="I love pizza")
```

### What Gets Saved

```json
{
  "program.predict": {
    "signature": {
      "instructions": "Classify the emotion...",
      "fields": {...}
    },
    "demos": [
      {
        "sentence": "I love this!",
        "emotion": "happy",
        "reasoning": "...",
        "augmented": true
      }
    ]
  }
}
```

---

## Comparison: Optimizer Outputs

### BootstrapFewShot

**Optimizes:** Demos only

**Result:**
```python
predictor.demos = [
    Example(sentence="I love this!", emotion="happy", reasoning="..."),
    Example(sentence="This is terrible.", emotion="sad", reasoning="...")
]
predictor.signature.instructions = "Classify the emotion of a sentence."  # Unchanged
```

### GEPA

**Optimizes:** Instructions only (typically)

**Result:**
```python
predictor.demos = []  # Usually empty
predictor.signature.instructions = """
# Instruction for the assistant

## Task Description
Given a sentence, classify the sentiment into: 'happy', 'sad', 'neutral', or 'frustration'.

## Task Requirements
1. Identify the sentiment
2. Classify into one of four emotions
3. Provide reasoning with analysis of structure, punctuation, emotional cues
...
"""  # Heavily optimized
```

### MIPRO

**Optimizes:** Both instructions AND demos

**Result:**
```python
predictor.demos = [demo1, demo2]  # Optimized selection
predictor.signature.instructions = "..."  # Optimized instructions
```

---

## Custom Optimization Patterns

### Pattern 1: Iterative Refinement

```python
def iterative_optimization(module, trainset, valset, metric, rounds=5):
    best_module = module
    best_score = evaluate(module, valset, metric)
    
    for round in range(rounds):
        # Try adding one more demo
        candidate = copy.deepcopy(best_module)
        new_demo = trainset[round]
        
        for name, predictor in candidate.named_predictors():
            predictor.demos.append(new_demo)
        
        score = evaluate(candidate, valset, metric)
        
        if score > best_score:
            best_module = candidate
            best_score = score
        else:
            break  # Stop if no improvement
    
    return best_module
```

### Pattern 2: A/B Testing

```python
def ab_test_demos(module, trainset, valset, metric):
    # Strategy A: Random selection
    module_a = copy.deepcopy(module)
    demos_a = random.sample(trainset, k=3)
    for name, predictor in module_a.named_predictors():
        predictor.demos = demos_a
    
    # Strategy B: Diverse selection
    module_b = copy.deepcopy(module)
    demos_b = select_diverse_demos(trainset, k=3)
    for name, predictor in module_b.named_predictors():
        predictor.demos = demos_b
    
    # Compare
    score_a = evaluate(module_a, valset, metric)
    score_b = evaluate(module_b, valset, metric)
    
    return module_a if score_a > score_b else module_b
```

### Pattern 3: Ensemble Optimization

```python
def ensemble_optimization(module_class, trainset, valset, metric):
    # Create multiple optimized versions
    optimizers = [
        BootstrapFewShot(max_bootstrapped_demos=2),
        BootstrapFewShot(max_bootstrapped_demos=4),
        LabeledFewShot(k=3)
    ]
    
    modules = []
    for opt in optimizers:
        optimized = opt.compile(module_class(), trainset=trainset)
        modules.append(optimized)
    
    # Evaluate and select best
    scores = [evaluate(m, valset, metric) for m in modules]
    best_idx = scores.index(max(scores))
    
    return modules[best_idx]
```

---

## Best Practices

### 1. Module Requirements

**✓ Correct:**
```python
class MyModule(dspy.Module):  # Inherit from dspy.Module
    def __init__(self):
        self.predictor = dspy.ChainOfThought(Signature)
    
    def forward(self, **kwargs):
        return self.predictor(**kwargs)

optimizer.compile(MyModule(), trainset=trainset)  # Works!
```

**✗ Incorrect:**
```python
predictor = dspy.Predict(Signature)  # Not a Module
optimizer.compile(predictor, trainset=trainset)  # Fails!
```

### 2. Metric Functions

**Simple metric:**
```python
def accuracy(example, pred, trace=None):
    return example.emotion == pred.emotion
```

**Metric with feedback (for GEPA):**
```python
def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    correct = example.emotion == pred.emotion
    
    if pred_name is None:
        return 1.0 if correct else 0.0
    
    feedback = f"Expected {example.emotion}, got {pred.emotion}"
    return dspy.Prediction(score=1.0 if correct else 0.0, feedback=feedback)
```

### 3. Train/Val Split

```python
# Good split
train = examples[:80]  # 80% for finding demos
val = examples[80:]    # 20% for validation

# For BootstrapFewShot
optimizer = BootstrapFewShot(max_bootstrapped_demos=5)
optimized = optimizer.compile(module, trainset=train)

# For GEPA (needs both)
optimizer = GEPA(metric=metric)
optimized = optimizer.compile(module, trainset=train, valset=val)
```

### 4. Selective Optimization

```python
class MultiStepModule(dspy.Module):
    def __init__(self):
        self.step1 = dspy.ChainOfThought(Signature1)
        self.step2 = dspy.Predict(Signature2)
    
    def forward(self, **kwargs):
        result1 = self.step1(**kwargs)
        result2 = self.step2(input=result1.output)
        return result2

# After optimization, inspect each predictor
for name, predictor in optimized.named_predictors():
    print(f"{name}: {len(predictor.demos)} demos")

# Output:
# step1.predict: 3 demos
# step2: 2 demos
```

### 5. Demo Quality Over Quantity

```python
# Bad: Too many demos (slow, may confuse model)
optimizer = BootstrapFewShot(max_bootstrapped_demos=20)  # ✗

# Good: Few high-quality demos
optimizer = BootstrapFewShot(
    max_bootstrapped_demos=3,  # ✓
    metric=strict_metric  # Only keep perfect examples
)
```

---

## Key Insights

1. **All optimizers set `predictor.demos`** - This is the universal interface
2. **Demos are (input, output) pairs** - Generated by running teacher on trainset
3. **Validation is critical** - Use metrics to filter bad demos
4. **Instructions vs Demos** - BootstrapFewShot optimizes demos, GEPA optimizes instructions, MIPRO does both
5. **Module requirement** - Student must be `dspy.Module`, not raw `dspy.Predict`
6. **Access via named_predictors()** - This reveals all optimizable components
7. **ChainOfThought wraps Predict** - Access internal predictor via `cot.predict.demos`
8. **Demos persist across sessions** - Save/load preserves optimized state
9. **Custom optimization is viable** - Grid search, A/B testing, ensemble methods all work
10. **Quality > Quantity** - 2-3 perfect demos often beat 10 mediocre ones

---

## Practical Example: Complete Workflow

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Define signature
class Emotion(dspy.Signature):
    """Classify the emotion of a sentence."""
    sentence = dspy.InputField()
    emotion = dspy.OutputField(desc="happy, sad, or neutral")

# 2. Create module
class EmotionClassifier(dspy.Module):
    def __init__(self):
        self.program = dspy.ChainOfThought(Emotion)
    
    def forward(self, sentence):
        return self.program(sentence=sentence)

# 3. Prepare data
trainset = [
    dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
    dspy.Example(sentence="This is terrible.", emotion="sad").with_inputs("sentence"),
    dspy.Example(sentence="It's okay.", emotion="neutral").with_inputs("sentence"),
]

# 4. Configure LM
lm = dspy.LM(model="ollama/llama3.2", temperature=0.0)
dspy.configure(lm=lm)

# 5. Optimize
optimizer = BootstrapFewShot(max_bootstrapped_demos=2)
optimized = optimizer.compile(EmotionClassifier(), trainset=trainset)

# 6. Inspect
for name, predictor in optimized.named_predictors():
    print(f"{name}: {len(predictor.demos)} demos")

# 7. Save
optimized.save("optimized_classifier.json")

# 8. Use
result = optimized(sentence="I love pizza")
print(result.emotion)  # "happy"
```

This workflow demonstrates the complete optimization cycle from definition to deployment