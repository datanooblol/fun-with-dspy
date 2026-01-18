# How to Create DSPy Demos: Best Practices

## Overview

Demos (few-shot examples) are critical for DSPy performance. This guide covers best practices for creating, selecting, and using demos effectively.

---

## What Are Demos?

Demos are example (input, output) pairs that show the LLM how to perform a task.

```python
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy",
    reasoning="The exclamation mark and word 'love' convey positive sentiment."
).with_inputs("sentence")  # Mark which fields are inputs
```

**Key components:**
- **Inputs**: Fields the model receives (marked with `.with_inputs()`)
- **Outputs**: Fields the model should generate
- **All fields**: Stored in the Example, but only inputs/outputs are shown in prompt

---

## Creating Demos: The Basics

### Method 1: Manual Creation

```python
# ✓ GOOD: Clear, correct example
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
).with_inputs("sentence")

# ✗ BAD: Missing .with_inputs()
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
)  # DSPy won't know which fields are inputs!
```

### Method 2: From Labeled Dataset

```python
# ✓ GOOD: Convert existing data
trainset = [
    dspy.Example(sentence=row['text'], emotion=row['label']).with_inputs("sentence")
    for row in dataset
]
```

### Method 3: Bootstrapping (Automatic)

```python
# ✓ GOOD: Let optimizer generate demos
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(max_bootstrapped_demos=3)
optimized = optimizer.compile(module, trainset=trainset)

# Demos are automatically created from successful teacher runs
```

---

## Best Practices: DO's

### ✓ DO: Keep Demos Short and Clear

```python
# ✓ GOOD: Concise, focused
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
).with_inputs("sentence")

# ✗ BAD: Too verbose
demo = dspy.Example(
    sentence="I really, truly, deeply love this amazing, wonderful, fantastic thing!",
    emotion="happy"
).with_inputs("sentence")
```

**Why:** Short demos are easier for LLM to process and less likely to confuse.

---

### ✓ DO: Cover Diverse Cases

```python
# ✓ GOOD: Diverse examples covering all classes
demos = [
    dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
    dspy.Example(sentence="This is terrible.", emotion="sad").with_inputs("sentence"),
    dspy.Example(sentence="It's okay.", emotion="neutral").with_inputs("sentence"),
]
```

**Why:** Diversity helps LLM understand the full range of expected outputs.

---

### ✓ DO: Use 2-5 Demos

```python
# ✓ GOOD: Small number of high-quality demos
optimizer = BootstrapFewShot(max_bootstrapped_demos=3)

# ✗ BAD: Too many demos
optimizer = BootstrapFewShot(max_bootstrapped_demos=20)
```

**Why:** 
- More demos = slower inference
- Too many demos can confuse the model
- 2-5 high-quality demos usually sufficient

---

### ✓ DO: Validate Demo Quality

```python
# ✓ GOOD: Use metric to filter demos
def strict_metric(example, pred, trace=None):
    # Only accept perfect matches
    return example.emotion == pred.emotion

optimizer = BootstrapFewShot(
    max_bootstrapped_demos=3,
    metric=strict_metric  # Only keep perfect examples
)
```

**Why:** Bad demos hurt performance more than no demos.

---

### ✓ DO: Match Demo Format to Task

```python
# ✓ GOOD: For ChainOfThought, include reasoning in demos
demo = dspy.Example(
    sentence="I love this!",
    reasoning="The exclamation mark indicates strong positive emotion.",
    emotion="happy"
).with_inputs("sentence")

# ✓ ALSO GOOD: For Predict, reasoning not needed
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
).with_inputs("sentence")
```

**Why:** ChainOfThought expects reasoning field; Predict doesn't.

**Note:** DSPy handles missing fields gracefully:
```
[[ ## reasoning ## ]]
Not supplied for this particular example.
```

---

### ✓ DO: Use Representative Examples

```python
# ✓ GOOD: Typical, common cases
demos = [
    dspy.Example(sentence="I love pizza", emotion="happy").with_inputs("sentence"),
    dspy.Example(sentence="I hate Mondays", emotion="sad").with_inputs("sentence"),
]

# ✗ BAD: Edge cases or unusual examples
demos = [
    dspy.Example(sentence="Antidisestablishmentarianism", emotion="neutral").with_inputs("sentence"),
    dspy.Example(sentence="🎉🎊🥳", emotion="happy").with_inputs("sentence"),
]
```

**Why:** Demos should represent typical inputs the model will see.

---

## Best Practices: DON'Ts

### ✗ DON'T: Use Contradictory Demos

```python
# ✗ BAD: Contradictory examples
demos = [
    dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
    dspy.Example(sentence="I love that!", emotion="sad").with_inputs("sentence"),  # Contradicts!
]
```

**Why:** Confuses the model about what "love" means.

---

### ✗ DON'T: Include Errors in Demos

```python
# ✗ BAD: Demo with wrong label
demo = dspy.Example(
    sentence="I love this!",
    emotion="sad"  # Wrong! Should be "happy"
).with_inputs("sentence")
```

**Why:** Model learns from demos, including mistakes.

---

### ✗ DON'T: Forget .with_inputs()

```python
# ✗ BAD: Missing .with_inputs()
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
)

# ✓ GOOD: Always specify inputs
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
).with_inputs("sentence")
```

**Why:** DSPy needs to know which fields are inputs vs outputs.

---

### ✗ DON'T: Mix Different Tasks

```python
# ✗ BAD: Demos for different tasks
demos = [
    dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
    dspy.Example(text="Translate to French", translation="Traduire en français").with_inputs("text"),
]
```

**Why:** Demos should all demonstrate the same task.

---

### ✗ DON'T: Use Demos with Missing Fields

```python
# ✗ BAD: Missing required output field
demo = dspy.Example(
    sentence="I love this!"
    # Missing emotion field!
).with_inputs("sentence")

# ✓ GOOD: All fields present
demo = dspy.Example(
    sentence="I love this!",
    emotion="happy"
).with_inputs("sentence")
```

**Why:** Incomplete demos don't show the full task.

---

### ✗ DON'T: Overfit to Demos

```python
# ✗ BAD: Using test data as demos
demos = testset[:3]  # Don't do this!

# ✓ GOOD: Separate train/test
demos = trainset[:3]
```

**Why:** Demos should not overlap with evaluation data.

---

## Demo Selection Strategies

### Strategy 1: Random Selection

```python
import random

# Simple random sampling
demos = random.sample(trainset, k=3)
```

**Pros:** Simple, unbiased  
**Cons:** May miss important cases

---

### Strategy 2: Diversity-Based Selection

```python
def select_diverse_demos(trainset, k=3):
    """Select demos covering different classes."""
    demos = []
    classes = set(ex.emotion for ex in trainset)
    
    for cls in classes:
        examples = [ex for ex in trainset if ex.emotion == cls]
        if examples:
            demos.append(random.choice(examples))
        if len(demos) >= k:
            break
    
    return demos
```

**Pros:** Ensures coverage of all classes  
**Cons:** May not work for imbalanced datasets

---

### Strategy 3: Similarity-Based Selection (KNN)

```python
def select_similar_demos(query, trainset, k=3):
    """Select demos most similar to query."""
    # Requires embedding function
    query_emb = embed(query)
    train_embs = [embed(ex.sentence) for ex in trainset]
    
    similarities = cosine_similarity([query_emb], train_embs)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]
    
    return [trainset[i] for i in top_k_idx]
```

**Pros:** Contextually relevant demos  
**Cons:** Requires embeddings, slower

---

### Strategy 4: Hard Example Mining

```python
def select_hard_examples(module, trainset, k=3):
    """Select examples the model gets wrong."""
    hard_examples = []
    
    for ex in trainset:
        pred = module(sentence=ex.sentence)
        if pred.emotion != ex.emotion:
            hard_examples.append(ex)
    
    return hard_examples[:k]
```

**Pros:** Targets model weaknesses  
**Cons:** May be too difficult for model

---

## Demo Quality Checklist

Before using demos, verify:

- [ ] All demos have `.with_inputs()` called
- [ ] All required output fields are present
- [ ] Labels are correct
- [ ] Examples are clear and concise
- [ ] Demos cover diverse cases
- [ ] No contradictions between demos
- [ ] Demo format matches predictor type (Predict vs ChainOfThought)
- [ ] Number of demos is reasonable (2-5)
- [ ] Demos don't overlap with test set

---

## Common Pitfalls

### Pitfall 1: Demo Contamination

**Problem:** Using conversation history as demos

```python
# ✗ BAD: Demos contain actual conversation context
demos = [
    dspy.Example(
        conversation_context="USER: Hi, my name is Alice\nASSISTANT: Hello Alice!",
        user_message="What's my name?",
        response="Your name is Alice."
    ).with_inputs("conversation_context", "user_message")
]
```

**Why it's bad:** LLM may confuse demo context with actual conversation context.

**Solution:** Use generic examples in demos, not actual conversation snippets.

---

### Pitfall 2: Inconsistent Reasoning

**Problem:** Reasoning doesn't match output

```python
# ✗ BAD: Reasoning contradicts emotion
demo = dspy.Example(
    sentence="I love this!",
    reasoning="The sentence expresses negative sentiment.",
    emotion="happy"  # Contradicts reasoning!
).with_inputs("sentence")
```

**Solution:** Ensure reasoning logically leads to output.

---

### Pitfall 3: Too Specific Demos

**Problem:** Demos are too narrow

```python
# ✗ BAD: All demos about pizza
demos = [
    dspy.Example(sentence="I love pizza", emotion="happy").with_inputs("sentence"),
    dspy.Example(sentence="Pizza is great", emotion="happy").with_inputs("sentence"),
    dspy.Example(sentence="Best pizza ever", emotion="happy").with_inputs("sentence"),
]
```

**Solution:** Use diverse topics and phrasings.

---

## Testing Demo Effectiveness

### Method 1: Ablation Study

```python
# Test with different numbers of demos
for k in [0, 1, 2, 3, 5]:
    module = EmotionClassifier()
    
    if k > 0:
        for name, predictor in module.named_predictors():
            predictor.demos = trainset[:k]
    
    score = evaluate(module, testset, metric)
    print(f"k={k}: score={score}")
```

### Method 2: Demo Swapping

```python
# Test different demo sets
demo_sets = [
    trainset[:3],           # First 3
    trainset[-3:],          # Last 3
    random.sample(trainset, 3)  # Random 3
]

for i, demos in enumerate(demo_sets):
    module = EmotionClassifier()
    for name, predictor in module.named_predictors():
        predictor.demos = demos
    
    score = evaluate(module, testset, metric)
    print(f"Demo set {i}: score={score}")
```

---

## Key Insights

1. **Quality > Quantity** - 2 perfect demos beat 10 mediocre ones
2. **Diversity matters** - Cover different classes and phrasings
3. **Validate everything** - Use metrics to filter bad demos
4. **Keep it simple** - Short, clear examples work best
5. **Match the format** - ChainOfThought needs reasoning, Predict doesn't
6. **Avoid contamination** - Don't mix demo context with actual context
7. **Test systematically** - Use ablation studies to find optimal k
8. **Representative examples** - Use typical cases, not edge cases
9. **No contradictions** - Ensure demos are consistent
10. **Always use .with_inputs()** - DSPy needs to know input vs output fields

---

## Quick Reference

### Creating a Demo

```python
demo = dspy.Example(
    input_field="input value",
    output_field="output value"
).with_inputs("input_field")
```

### Setting Demos on Predictor

```python
# For dspy.Predict
predictor = dspy.Predict(Signature)
predictor.demos = [demo1, demo2, demo3]

# For dspy.ChainOfThought
cot = dspy.ChainOfThought(Signature)
cot.predict.demos = [demo1, demo2, demo3]
```

### Automatic Demo Generation

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(max_bootstrapped_demos=3)
optimized = optimizer.compile(module, trainset=trainset)
```

### Inspecting Demos

```python
for name, predictor in module.named_predictors():
    print(f"{name}: {len(predictor.demos)} demos")
    for i, demo in enumerate(predictor.demos):
        print(f"  Demo {i+1}: {demo}")
```
