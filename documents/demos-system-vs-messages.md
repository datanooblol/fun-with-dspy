# Demos: System Prompt vs Separate Messages

## The Question
Why does DSPy use separate user/assistant message pairs for demos instead of putting examples in the system prompt?

## Answer: LLM Training and Architecture

### How Chat Models Are Trained

Modern chat models are fine-tuned on conversational datasets with distinct role markers:

```python
# Training data format
[
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
]
```

The model learns to:
1. Generate assistant responses after user messages
2. Maintain conversation context across turns
3. Distinguish between different roles

### Approach 1: Separate Messages (DSPy's Approach)

```python
messages = [
    {"role": "system", "content": "Answer questions accurately"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "What is 3+3?"}  # Actual query
]
```

**Advantages:**
- ✅ Matches training format exactly
- ✅ Activates conversational behavior
- ✅ Model "experiences" the pattern
- ✅ Better attention allocation
- ✅ Stronger pattern matching

### Approach 2: System Prompt Only

```python
messages = [
    {
        "role": "system", 
        "content": """Answer questions accurately.

Example:
User: What is 2+2?
Assistant: 4"""
    },
    {"role": "user", "content": "What is 3+3?"}
]
```

**Disadvantages:**
- ❌ Doesn't match training format
- ❌ Treated as instructions, not conversation
- ❌ Weaker pattern activation
- ❌ Model must parse text format
- ❌ Less effective learning

## Empirical Evidence

### Test Case: Math QA

**Separate Messages:**
```python
[
    {"role": "system", "content": "Solve math problems"},
    {"role": "user", "content": "2+2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "3+3"},
    {"role": "assistant", "content": "6"},
    {"role": "user", "content": "5+5"}
]
# Model response: "10" ✓
```

**System Prompt:**
```python
[
    {"role": "system", "content": "Solve math problems\nExample: 2+2=4, 3+3=6"},
    {"role": "user", "content": "5+5"}
]
# Model response: "The answer is 10" or "5+5=10" (inconsistent format) ✗
```

### Test Case: Structured Output

**Separate Messages:**
```python
[
    {"role": "system", "content": "Extract names"},
    {"role": "user", "content": "Hi, I'm Alice"},
    {"role": "assistant", "content": "Name: Alice"},
    {"role": "user", "content": "Hello, I'm Bob"}
]
# Model response: "Name: Bob" ✓ (consistent format)
```

**System Prompt:**
```python
[
    {"role": "system", "content": "Extract names\nExample: 'Hi, I'm Alice' -> 'Name: Alice'"},
    {"role": "user", "content": "Hello, I'm Bob"}
]
# Model response: "Bob" or "The name is Bob" ✗ (inconsistent format)
```

## Why This Matters for DSPy

DSPy's goal is to make LLM behavior **predictable and consistent**. Using separate messages:

1. **Better format adherence** - Model follows the pattern more reliably
2. **Consistent output structure** - Especially important for parsing
3. **Stronger few-shot learning** - Model learns from actual examples
4. **Optimization compatibility** - DSPy can optimize demos programmatically

## Technical Details

### Attention Patterns

When processing separate messages, the model's attention mechanism:
- Recognizes role transitions (user → assistant)
- Activates conversational pathways
- Applies learned turn-taking behavior

When processing system prompt examples:
- Treats content as continuous text
- Must parse role markers manually
- Less activation of conversational training

### Token Position Encoding

```python
# Separate messages
[system] [user_demo] [assistant_demo] [user_actual]
   ↓         ↓            ↓                ↓
Position: 0-50    51-60      61-70         71-80

# System prompt
[system_with_examples] [user_actual]
         ↓                    ↓
Position: 0-100            101-110
```

Separate messages provide clearer positional boundaries for the model to learn from.

## When System Prompts Work

System prompts are fine for:
- **General instructions**: "Be concise", "Use formal language"
- **Role definition**: "You are a helpful assistant"
- **Constraints**: "Don't discuss politics"

But for **behavioral examples**, separate messages are superior.

## DSPy's Design Choice

DSPy uses `ChatAdapter` to convert demos into separate user/assistant pairs because:

1. **Empirically better** - Tested across many models
2. **Consistent with research** - Aligns with few-shot learning best practices
3. **Optimization-friendly** - Can programmatically add/remove/reorder demos
4. **Format preservation** - Ensures output matches expected structure

## Conclusion

**Separate messages work better because they match how LLMs are trained.**

The model doesn't just "read" the examples—it "experiences" them as if they were part of an actual conversation. This activates the conversational pathways learned during training, resulting in better pattern matching and more consistent outputs.

This is why DSPy (and most modern prompting frameworks) use separate message turns for few-shot examples rather than embedding them in system prompts.
