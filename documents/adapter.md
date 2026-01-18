# DSPy ChatAdapter Documentation

## Overview

`ChatAdapter` is DSPy's mechanism for converting `Signature` definitions into structured chat messages that LLMs can understand. It transforms your high-level signature (input/output fields) into a formatted prompt with system instructions, field markers, and optional few-shot examples.

## How ChatAdapter Works

### Core Concept

When you use `dspy.Predict(YourSignature)`, DSPy internally uses `ChatAdapter` to:
1. Generate a **system message** describing the task structure
2. Add **demo messages** (if provided) as user/assistant pairs
3. Format the **actual input** as the final user message

### Message Structure

```
[System Message]
├─ Field descriptions (inputs + outputs)
├─ Field structure template with [[ ## field ## ]] markers
└─ Task objective (from signature docstring)

[Demo 1 - User Message]
└─ Input fields with values

[Demo 1 - Assistant Message]
└─ Output fields with values

[Demo 2 - User/Assistant...]
...

[Actual Input - User Message]
└─ Your actual input values + output format reminder
```

## ChatAdapter Methods

### 1. `format_field_description(signature)`
Creates a list of input and output fields with descriptions.

```python
from dspy.adapters.chat_adapter import ChatAdapter

class QA(dspy.Signature):
    """Answer questions accurately"""
    question: str = dspy.InputField(desc="The question")
    answer: str = dspy.OutputField(desc="The answer")

adapter = ChatAdapter()
print(adapter.format_field_description(QA))
```

**Output:**
```
Your input fields are:
1. `question` (str): The question
Your output fields are:
1. `answer` (str): The answer
```

### 2. `format_field_structure(signature)`
Creates the template structure with `[[ ## field ## ]]` markers.

```python
print(adapter.format_field_structure(QA))
```

**Output:**
```
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
```

### 3. `format_task_description(signature)`
Extracts the objective from the signature's docstring.

```python
print(adapter.format_task_description(QA))
```

**Output:**
```
In adhering to this structure, your objective is: 
        Answer questions accurately
```

### 4. `format_user_message_content(signature, inputs, main_request=True)`
Formats input values into the field structure.

```python
user_msg = adapter.format_user_message_content(
    signature=QA,
    inputs={"question": "What is 2+2?"},
    main_request=True  # Adds output format reminder
)
print(user_msg)
```

**Output:**
```
[[ ## question ## ]]
What is 2+2?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
```

### 5. `format(signature, demos, inputs)`
**The main method** - generates complete message list including system prompt, demos, and actual input.

## Examples

### Example 1: Without Demos (Zero-Shot)

```python
from dspy.adapters.chat_adapter import ChatAdapter
import dspy

class QA(dspy.Signature):
    """Answer questions accurately"""
    question: str = dspy.InputField(desc="The question")
    answer: str = dspy.OutputField(desc="The answer")

adapter = ChatAdapter()

# Generate messages without demos
messages = adapter.format(
    signature=QA,
    demos=[],  # No demos
    inputs={"question": "What is the capital of Thailand?"}
)

# Result: 2 messages
# messages[0] = system message with structure
# messages[1] = user message with actual input
```

**Generated Messages:**
```python
[
    {
        "role": "system",
        "content": """Your input fields are:
1. `question` (str): The question
Your output fields are:
1. `answer` (str): The answer
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Answer questions accurately"""
    },
    {
        "role": "user",
        "content": """[[ ## question ## ]]
What is the capital of Thailand?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."""
    }
]
```

### Example 2: With Demos (Few-Shot)

```python
from dspy.adapters.chat_adapter import ChatAdapter
import dspy

class QA(dspy.Signature):
    """Answer questions accurately"""
    question: str = dspy.InputField(desc="The question")
    answer: str = dspy.OutputField(desc="The answer")

# Create demos
demos = [
    dspy.Example(
        question="What is 2+2?",
        answer="4"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What color is the sky?",
        answer="Blue"
    ).with_inputs("question"),
]

adapter = ChatAdapter()

# Generate messages with demos
messages = adapter.format(
    signature=QA,
    demos=demos,
    inputs={"question": "What is the capital of Thailand?"}
)

# Result: 7 messages
# messages[0] = system
# messages[1] = demo1 user
# messages[2] = demo1 assistant
# messages[3] = demo2 user
# messages[4] = demo2 assistant
# messages[5] = actual input user
```

**Generated Messages:**
```python
[
    {
        "role": "system",
        "content": "Your input fields are:\n1. `question` (str): The question\n..."
    },
    {
        "role": "user",
        "content": "[[ ## question ## ]]\nWhat is 2+2?"
    },
    {
        "role": "assistant",
        "content": "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]\n"
    },
    {
        "role": "user",
        "content": "[[ ## question ## ]]\nWhat color is the sky?"
    },
    {
        "role": "assistant",
        "content": "[[ ## answer ## ]]\nBlue\n\n[[ ## completed ## ]]\n"
    },
    {
        "role": "user",
        "content": "[[ ## question ## ]]\nWhat is the capital of Thailand?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."
    }
]
```

### Example 3: Chatbot with Demos

```python
import dspy

class ChatSignature(dspy.Signature):
    """Respond to user in a conversation"""
    chat_history: str = dspy.InputField(desc="Previous chat history")
    user_message: str = dspy.InputField(desc="Current user message")
    response: str = dspy.OutputField(desc="Assistant's response")

# Create demos showing conversation patterns
demos = [
    dspy.Example(
        chat_history="No previous messages",
        user_message="Hi, my name is Alice",
        response="Hello Alice! Nice to meet you. How can I help you today?"
    ).with_inputs("chat_history", "user_message"),
    
    dspy.Example(
        chat_history="USER: Hi, my name is Alice\nASSISTANT: Hello Alice! Nice to meet you. How can I help you today?",
        user_message="What's my name?",
        response="Your name is Alice."
    ).with_inputs("chat_history", "user_message"),
]

# Use in ChatAgent
class ChatAgent(dspy.Module):
    def __init__(self, demos=None):
        self.program = dspy.Predict(ChatSignature)
        if demos:
            self.program.demos = demos
        self.chat_history = []

    def forward(self, message: str):
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.chat_history]) if self.chat_history else "No previous messages"
        
        result = self.program(chat_history=history_str, user_message=message)
        
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": result.response})
        
        return result.response

# Create agent with demos
agent = ChatAgent(demos=demos)
```

**What gets sent to LLM:**
```python
[
    {"role": "system", "content": "Your input fields are:\n1. `chat_history` (str): Previous chat history\n2. `user_message` (str): Current user message\n..."},
    {"role": "user", "content": "[[ ## chat_history ## ]]\nNo previous messages\n\n[[ ## user_message ## ]]\nHi, my name is Alice"},
    {"role": "assistant", "content": "[[ ## response ## ]]\nHello Alice! Nice to meet you. How can I help you today?\n\n[[ ## completed ## ]]\n"},
    {"role": "user", "content": "[[ ## chat_history ## ]]\nUSER: Hi, my name is Alice\nASSISTANT: Hello Alice! Nice to meet you. How can I help you today?\n\n[[ ## user_message ## ]]\nWhat's my name?"},
    {"role": "assistant", "content": "[[ ## response ## ]]\nYour name is Alice.\n\n[[ ## completed ## ]]\n"},
    {"role": "user", "content": "[[ ## chat_history ## ]]\nNo previous messages\n\n[[ ## user_message ## ]]\nMy name is Bank\n\nRespond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."}
]
```

## Key Insights

1. **Demos use message turns**: Each demo becomes a user/assistant message pair, teaching the LLM the expected input/output pattern
2. **Field markers are consistent**: All messages (demos and actual input) use the same `[[ ## field ## ]]` format
3. **System message is shared**: One system message describes the structure for all interactions
4. **Last message is special**: The final user message includes a reminder to respond with the output format
5. **Demos teach behavior**: Use demos to show the LLM how to handle edge cases, formatting, or specific response styles

## When to Use Demos

- **Zero-shot (no demos)**: When the task is simple and self-explanatory
- **Few-shot (with demos)**: When you need to:
  - Show specific output formatting
  - Demonstrate handling of edge cases
  - Teach conversation patterns (like memory in chatbots)
  - Improve consistency and quality

## Inspecting Generated Messages

To see what DSPy sends to your LLM:

```python
# After making a prediction
lm = dspy.settings.lm  # or your configured LM
last_call = lm.history[-1]

# View system message
print(last_call['messages'][0]['content'])

# View all messages
for msg in last_call['messages']:
    print(f"{msg['role'].upper()}:")
    print(msg['content'])
    print("=" * 50)
```
