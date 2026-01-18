# dspy.Signature vs dspy.ChainOfThought

## Overview

This document explains the differences between `dspy.Predict(Signature)` and `dspy.ChainOfThought(Signature)` based on experiments in `04_signature_cot.ipynb`.

Both are DSPy predictors that use signatures to structure LLM interactions, but they differ in how they prompt the LLM to generate responses.

---

## Core Difference

**dspy.Predict**: Direct prediction from inputs to outputs
**dspy.ChainOfThought**: Adds explicit reasoning step before generating outputs

---

## Implementation Details

### dspy.Predict(Signature)

```python
qa_zero_shot = dspy.Predict(ChatSignature)
```

**What it does:**
- Uses signature as-is
- Generates output fields directly from input fields
- No intermediate reasoning step

**Signature structure:**
```
ChatSignature(
    conversation_context, user_message -> response
    instructions='Use the conversation_context to answer questions about the user.'
)
```

**Output fields:**
1. `response` (str): Answer using conversation_context

---

### dspy.ChainOfThought(Signature)

```python
qa_cot = dspy.ChainOfThought(ChatSignature)
```

**What it does:**
- Modifies signature by prepending `reasoning` field to output_fields
- Forces LLM to generate reasoning before final response
- Uses `signature.prepend()` method internally

**Modified signature structure:**
```
StringSignature(
    conversation_context, user_message -> reasoning, response
    instructions='Use the conversation_context to answer questions about the user.'
)
```

**Output fields:**
1. `reasoning` (str): Step-by-step thinking process
2. `response` (str): Answer using conversation_context

---

## How ChainOfThought Works Internally

From `dspy/predict/chain_of_thought.py`:

```python
class ChainOfThought(Predict):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__(signature, **config)
        
        # Create reasoning field
        self.activated = activated
        signature = self.signature
        *keys, last_key = signature.output_fields.keys()
        
        # Prepend reasoning to output fields
        DEFAULT_RATIONALE_TYPE = dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${reasoning}"
        )
        
        rationale_type = rationale_type or DEFAULT_RATIONALE_TYPE
        self.extended_signature = self.signature.prepend("reasoning", rationale_type, type_=str)
```

**Key mechanism:**
- `signature.prepend("reasoning", field_spec, type_=str)` creates NEW signature
- Reasoning field is added as FIRST output field
- Original signature remains unchanged
- `prepend()` is custom DSPy method (not Python dict method)

---

## Prompt Structure Comparison

### dspy.Predict Prompt

**System message:**
```
Your input fields are:
1. `conversation_context` (str): All previous messages in this conversation
2. `user_message` (str): Current question
Your output fields are:
1. `response` (str): Answer using conversation_context
All interactions will be structured in the following way...

[[ ## conversation_context ## ]]
{conversation_context}

[[ ## user_message ## ]]
{user_message}

[[ ## response ## ]]
{response}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Use the conversation_context to answer questions about the user.
```

**User message (actual query):**
```
[[ ## conversation_context ## ]]

USER: My name is Bank. Nice to meet you.
ASSISTANT: Hello Bank! It's nice to meet you.
...

[[ ## user_message ## ]]
What's my name?

Respond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, 
and then ending with the marker for `[[ ## completed ## ]]`.
```

**LLM Response:**
```
[[ ## response ## ]]
Your name is Bank.

[[ ## completed ## ]]
```

---

### dspy.ChainOfThought Prompt

**System message:**
```
Your input fields are:
1. `conversation_context` (str): All previous messages in this conversation
2. `user_message` (str): Current question
Your output fields are:
1. `reasoning` (str): 
2. `response` (str): Answer using conversation_context
All interactions will be structured in the following way...

[[ ## conversation_context ## ]]
{conversation_context}

[[ ## user_message ## ]]
{user_message}

[[ ## reasoning ## ]]
{reasoning}

[[ ## response ## ]]
{response}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Use the conversation_context to answer questions about the user.
```

**User message (actual query):**
```
[[ ## conversation_context ## ]]

USER: My name is Bank. Nice to meet you.
ASSISTANT: Hello Bank! It's nice to meet you.
...

[[ ## user_message ## ]]
What's my name?

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, 
then `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
```

**LLM Response:**
```
[[ ## reasoning ## ]]
Based on the conversation context, the user has previously introduced themselves as "Bank". 
Therefore, the user's name is Bank.

[[ ## response ## ]]
Your name is Bank.

[[ ## completed ## ]]
```

---

## Demo Handling with ChainOfThought

### Important Discovery

When using demos with ChainOfThought, DSPy handles missing `reasoning` field gracefully:

**Demo without reasoning:**
```python
dspy.Example(
    conversation_context="No previous messages",
    user_message="Hi, my name is Alice",
    response="Hello Alice! Nice to meet you."
).with_inputs("conversation_context", "user_message")
```

**How it appears in prompt:**
```
This is an example of the task, though some input or output fields are not supplied.

[[ ## conversation_context ## ]]
No previous messages

[[ ## user_message ## ]]
Hi, my name is Alice

[[ ## reasoning ## ]]
Not supplied for this particular example. 

[[ ## response ## ]]
Hello Alice! Nice to meet you.

[[ ## completed ## ]]
```

**Key insight:** DSPy automatically fills missing output fields with "Not supplied for this particular example." This allows:
- Using same demos for both Predict and ChainOfThought
- No need to create separate demo sets with reasoning
- LLM still understands to generate reasoning for actual queries

---

## Accessing Internal Predictor

### dspy.Predict
```python
qa_zero_shot = dspy.Predict(ChatSignature)
qa_zero_shot.demos = demos  # Direct access
```

### dspy.ChainOfThought
```python
qa_cot = dspy.ChainOfThought(ChatSignature)
qa_cot.predict.demos = demos  # Access via .predict attribute
```

**Why the difference?**
- ChainOfThought wraps a Predict instance internally
- The wrapped predictor is stored in `self.predict`
- To set demos, you must access the internal predictor

---

## Output Comparison

### dspy.Predict Output
```python
Prediction(
    response='Your name is Bank.'
)
```

**Token usage:** 593 prompt + 19 completion = 612 total

---

### dspy.ChainOfThought Output
```python
Prediction(
    reasoning='Based on the conversation context, the user has previously introduced themselves as "Bank". Therefore, the user\'s name is Bank.',
    response='Your name is Bank.'
)
```

**Token usage:** 714 prompt + 51 completion = 765 total

**Cost difference:**
- More prompt tokens (reasoning field in system message + demos)
- More completion tokens (reasoning + response vs just response)
- ~25% increase in total tokens for this example

---

## When to Use Each

### Use dspy.Predict when:
- Task is straightforward and doesn't require explicit reasoning
- Token efficiency is critical
- You want faster responses
- The LLM can generate correct outputs directly

### Use dspy.ChainOfThought when:
- Task requires multi-step reasoning
- You want to see the LLM's thought process
- Accuracy is more important than speed/cost
- Complex queries benefit from structured thinking
- Debugging why LLM made certain decisions

---

## Key Insights

1. **ChainOfThought is a wrapper**: It creates a modified signature and uses Predict internally
2. **prepend() is custom DSPy method**: Not a standard Python dict method, operates on output_fields OrderedDict
3. **Demo compatibility**: Same demos work for both Predict and ChainOfThought due to automatic field filling
4. **Token cost**: ChainOfThought uses ~20-30% more tokens due to reasoning field
5. **Transparency**: ChainOfThought provides visibility into LLM's reasoning process
6. **Non-destructive**: Original signature remains unchanged, new signature is created

---

## Design Philosophy

DSPy's approach demonstrates excellent library design:

1. **Composability**: ChainOfThought builds on Predict, not reimplementing everything
2. **Flexibility**: Same signature works for both predictors
3. **Transparency**: Reasoning is explicit, not hidden in system prompt
4. **Backward compatibility**: Demos without reasoning still work
5. **Minimal API surface**: Simple interface hides complex prompt engineering

This design opens possibilities for:
- Custom predictors that modify signatures in different ways
- Hybrid approaches (reasoning only for complex queries)
- Dynamic switching between Predict and ChainOfThought based on query complexity
- Custom reasoning formats (e.g., structured reasoning, multi-step planning)

---

## Practical Example from Notebook

**Setup:**
```python
class ChatSignature(dspy.Signature):
    """Use the conversation_context to answer questions about the user."""
    conversation_context: str = dspy.InputField(desc="All previous messages in this conversation")
    user_message: str = dspy.InputField(desc="Current question")
    response: str = dspy.OutputField(desc="Answer using conversation_context")

qa_zero_shot = dspy.Predict(ChatSignature)
qa_cot = dspy.ChainOfThought(ChatSignature)
```

**Query:**
```python
params = {
    "conversation_context": "USER: My name is Bank...",
    "user_message": "What's my name?"
}
```

**Results:**
- Both return: "Your name is Bank."
- ChainOfThought additionally shows reasoning process
- ChainOfThought uses more tokens but provides transparency
