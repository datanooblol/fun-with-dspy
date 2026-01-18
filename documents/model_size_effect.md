# Model Size Effects on DSPy Structured Prompting

## Overview

DSPy uses structured prompting with field markers like `[[ ## field ## ]]` to format inputs and outputs. This structured approach works differently across models of varying sizes and capabilities.

## Key Finding

**Not all models handle DSPy's structured prompting equally well.** Smaller or less capable models may struggle with the `[[ ## field ## ]]` format and meta-instructions about using fields.

## Empirical Results

### Test Case: Chatbot Memory

**Task**: Remember user's name from chat history and recall it when asked.

**Prompt Structure**:
```
[[ ## chat_history ## ]]
USER: My name is Bank. Nice to meet you.
ASSISTANT: Hello Bank! It's nice to meet you.
...

[[ ## user_message ## ]]
What's my name?
```

### Results by Model

| Model | Size/Tier | Result | Notes |
|-------|-----------|--------|-------|
| **Llama 3.2 Vision** | 11B | ✅ Success | Correctly retrieved "Bank" from chat_history |
| **AWS Nova Micro** | Smallest | ❌ Failed | Said "I don't have that information" then contradicted itself |
| **AWS Nova Lite** | Medium | ✅ Success | Correctly retrieved "Bank" from chat_history |
| **AWS Nova Pro** | Large | ✅ Success (expected) | Should handle complex structures well |

## Why Smaller Models Struggle

### 1. Complex Prompt Structure
DSPy's field markers add meta-structure:
```
[[ ## field_name ## ]]
{field_content}
```

Smaller models may:
- Treat field markers as literal text
- Confuse field content with actual conversation
- Struggle to understand meta-instructions

### 2. Instruction Following
DSPy signatures include instructions like:
```
"Respond with the corresponding output fields, starting with the field `[[ ## response ## ]]`"
```

Smaller models may not follow complex multi-step instructions reliably.

### 3. Context Separation
Models need to understand:
- System prompt = instructions
- Demo messages = examples
- Field content = data to use
- Actual input = current query

Smaller models may blur these boundaries.

## Recommendations

### For Production Use

**If using smaller models (< 10B parameters or "micro" tier):**

1. **Simplify signatures**
   ```python
   # Instead of:
   class ChatSignature(dspy.Signature):
       """Respond based only on the provided chat_history. Examples are for format only."""
       chat_history: str = dspy.InputField(desc="Actual conversation history")
       user_message: str = dspy.InputField(desc="Current user message")
       response: str = dspy.OutputField(desc="Assistant's response")
   
   # Use:
   class ChatSignature(dspy.Signature):
       """Answer questions."""
       history: str = dspy.InputField()
       question: str = dspy.InputField()
       answer: str = dspy.OutputField()
   ```

2. **Remove or minimize demos**
   ```python
   # Smaller models may get confused by demos
   ca = ChatAgent(demos=None)  # No demos
   ```

3. **Use clearer field names**
   - Avoid: `chat_history`, `conversation_history`
   - Prefer: `history`, `context`, `previous_messages`

4. **Test thoroughly**
   - Verify the model handles your specific use case
   - Check edge cases (empty history, long history, etc.)

**If using medium/large models (10B+ parameters or "lite"/"pro" tier):**

- Full DSPy features work well
- Use demos for few-shot learning
- Complex signatures are fine
- Multiple fields are handled correctly

### Model Selection Guide

**For DSPy with structured prompting:**

| Use Case | Minimum Recommended | Optimal |
|----------|-------------------|---------|
| Simple QA | 7B+ (Lite tier) | 13B+ (Pro tier) |
| Chatbot with memory | 10B+ (Lite tier) | 13B+ (Pro tier) |
| Complex reasoning | 13B+ (Pro tier) | 70B+ (Ultra tier) |
| Production critical | 13B+ (Pro tier) | Claude/GPT-4 class |

### AWS Bedrock Specific

**Nova Model Tiers:**
- **Nova Micro**: ❌ Not recommended for DSPy (struggles with structured prompts)
- **Nova Lite**: ✅ Good for DSPy (handles structure well)
- **Nova Pro**: ✅ Excellent for DSPy (full capability)

**Alternative Bedrock Models:**
- **Claude 3.5 Sonnet**: ✅ Excellent (best for complex tasks)
- **Claude 3 Haiku**: ✅ Good (fast, handles structure well)

## Debugging Model Issues

### Symptom: Model ignores field content

**Example**: Model says "I don't have that information" when info is clearly in a field.

**Diagnosis**: Model doesn't understand field structure.

**Solutions**:
1. Use larger model
2. Simplify signature
3. Remove demos
4. Make field names more explicit

### Symptom: Model contradicts itself

**Example**: "I don't have your name. However, your name is Bank."

**Diagnosis**: Model treats field content as separate from "this conversation."

**Solutions**:
1. Use larger model
2. Change signature docstring to be more explicit:
   ```python
   """Use the information provided in the history field to answer questions."""
   ```

### Symptom: Model doesn't follow output format

**Example**: Model doesn't use `[[ ## response ## ]]` markers.

**Diagnosis**: Model doesn't understand meta-instructions.

**Solutions**:
1. Use larger model
2. Add demos showing correct format
3. Simplify output requirements

## Testing Your Model

Quick test to verify model handles DSPy structure:

```python
import dspy

class MemoryTest(dspy.Signature):
    """Answer using the context provided."""
    context: str = dspy.InputField(desc="Information to use")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Answer from context")

# Configure your model
dspy.configure(lm=your_lm)

# Test
predictor = dspy.Predict(MemoryTest)
result = predictor(
    context="The user's name is Alice. She likes pizza.",
    question="What is the user's name?"
)

print(result.answer)
# Expected: "Alice" or "The user's name is Alice"
# Bad: "I don't have that information" or contradictory response
```

If the test fails, your model may struggle with DSPy's structured prompting.

## Summary

**Key Takeaway**: DSPy's structured prompting requires models with sufficient capability to understand:
1. Field markers and their meaning
2. Meta-instructions about using fields
3. Separation between examples and actual input
4. Complex multi-step instructions

**Rule of Thumb**:
- **Micro/Tiny models**: May struggle, simplify or avoid DSPy
- **Small models (7-10B)**: Test carefully, simplify if needed
- **Medium models (10-20B)**: Generally work well
- **Large models (20B+)**: Full DSPy capability

**For AWS Bedrock**: Start with Nova Lite or higher for reliable DSPy usage. Nova Micro is too small for structured prompting.
