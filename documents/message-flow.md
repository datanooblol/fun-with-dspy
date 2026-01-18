# DSPy Message Flow: From Signature to LLM

## Overview

Understanding how DSPy constructs and sends messages is crucial for adapting DriverLM to different LLM providers. This document traces the complete journey from your high-level signature to the actual API call.

## The Complete Flow

```
Your Code: ca(message="Hi")
    ↓
ChatAgent.forward(message="Hi")
    ↓
self.program(chat_history="No previous messages", user_message="Hi")
    ↓
[DSPy Internal: dspy.Predict]
    ↓
ChatAdapter.format(signature=ChatSignature, demos=demos, inputs={...})
    ↓
[Generates complete messages list with system + demos + input]
    ↓
lm(messages=[...])  # Your configured LM
    ↓
DriverLM.__call__(messages=[...])
    ↓
DriverLM.forward(messages=[...])
    ↓
request_fn(messages=[...], temperature=0.0, ...)
    ↓
[Your LLM Provider API]
```

## Step-by-Step Breakdown

### Step 1: Your Code

```python
class ChatAgent(dspy.Module):
    def __init__(self, demos=None):
        self.program = dspy.Predict(ChatSignature)
        if demos:
            self.program.demos = demos

    def forward(self, message: str):
        chat_history = "No previous messages"
        response = self.program(chat_history=chat_history, user_message=message)
        return response.response

ca = ChatAgent(demos=demos)
ca(message="Hi")
```

**What you provide:**
- Field values: `chat_history="No previous messages"`, `user_message="Hi"`
- Demos (optional): List of `dspy.Example` objects

**What you DON'T provide:**
- System prompt (DSPy generates from signature)
- Message structure (DSPy handles formatting)

### Step 2: DSPy.Predict Processing

When you call `self.program(chat_history=..., user_message=...)`:

1. DSPy's `Predict` class receives your field values
2. It calls `ChatAdapter.format()` internally
3. ChatAdapter constructs the complete messages list

### Step 3: ChatAdapter Message Construction

```python
# ChatAdapter.format() generates:
messages = [
    # System message (from signature)
    {
        "role": "system",
        "content": """Your input fields are:
1. `chat_history` (str): Actual conversation history
2. `user_message` (str): Current user message
Your output fields are:
1. `response` (str): Assistant's response
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## chat_history ## ]]
{chat_history}

[[ ## user_message ## ]]
{user_message}

[[ ## response ## ]]
{response}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Respond based only on the provided chat_history. Examples are for format only."""
    },
    
    # Demo 1 - User
    {
        "role": "user",
        "content": "[[ ## chat_history ## ]]\nNo previous messages\n\n[[ ## user_message ## ]]\nHi, my name is Alice"
    },
    
    # Demo 1 - Assistant
    {
        "role": "assistant",
        "content": "[[ ## response ## ]]\nHello Alice! Nice to meet you. How can I help you today?\n\n[[ ## completed ## ]]\n"
    },
    
    # Demo 2 - User
    {
        "role": "user",
        "content": "[[ ## chat_history ## ]]\nUSER: Hi, my name is Alice\nASSISTANT: Hello Alice!\n\n[[ ## user_message ## ]]\nWhat's my name?"
    },
    
    # Demo 2 - Assistant
    {
        "role": "assistant",
        "content": "[[ ## response ## ]]\nYour name is Alice.\n\n[[ ## completed ## ]]\n"
    },
    
    # Actual input - User
    {
        "role": "user",
        "content": "[[ ## chat_history ## ]]\nNo previous messages\n\n[[ ## user_message ## ]]\nHi\n\nRespond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."
    }
]
```

### Step 4: DriverLM Receives Messages

DSPy calls your configured LM with the complete messages list:

```python
# In DriverLM.forward()
def forward(self, prompt=None, messages=None, **kwargs):
    # messages is already fully constructed by DSPy
    messages = messages or [{"role": "user", "content": prompt}]
    
    # Build request
    request = dict(
        model="custom",
        messages=messages,  # Complete messages list from ChatAdapter
        **kwargs
    )
    
    # Call your request_fn
    result = self._cached_forward(request=request)
    return result
```

### Step 5: Your request_fn Receives Messages

```python
def ollama_request_fn(messages: list[dict[str, Any]], temperature: float = 0.0, **kwargs) -> dict:
    # messages is the complete list from ChatAdapter
    # You just pass it to your LLM provider
    
    response = ollama_client.post(
        'http://localhost:11434/api/chat',
        json={
            "model": "llama3.2-vision:11b",
            "messages": messages,  # Pass through directly
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    return response.json()
```

## Key Insights

### 1. You Never Construct System Prompts

DSPy automatically generates system prompts from:
- Signature docstring → Task objective
- InputField descriptions → Input field list
- OutputField descriptions → Output field list
- Field structure → `[[ ## field ## ]]` template

### 2. Demos Are Pre-Formatted

When you provide demos, DSPy:
- Converts each demo into user/assistant message pairs
- Formats them with `[[ ## field ## ]]` markers
- Inserts them between system prompt and actual input

### 3. Your request_fn Is Provider-Agnostic

The signature is always:
```python
def request_fn(messages: list[dict[str, Any]], temperature: float, **kwargs) -> dict:
    # messages is already formatted
    # Just adapt to your provider's API
    pass
```

### 4. Message Format Is OpenAI-Compatible

DSPy uses OpenAI's message format:
```python
{"role": "system" | "user" | "assistant", "content": str}
```

Most providers accept this format or have simple conversions.

## Adapting to Different Providers

### Pattern 1: Direct Pass-Through (Ollama, OpenAI-compatible)

```python
def request_fn(messages, temperature=0.0, **kwargs):
    # Provider accepts OpenAI format directly
    return provider_client.chat(
        model="model-name",
        messages=messages,  # Pass through
        temperature=temperature
    )
```

### Pattern 2: Format Conversion (AWS Bedrock)

```python
def bedrock_request_fn(messages, temperature=0.0, **kwargs):
    # Bedrock separates system from conversation
    system_messages = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    conversation = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    
    return client.converse(
        modelId="model-id",
        system=system_messages,
        messages=conversation,
        inferenceConfig={"temperature": temperature}
    )
```

### Pattern 3: Custom Protocol (Anthropic Direct)

```python
def anthropic_request_fn(messages, temperature=0.0, **kwargs):
    # Anthropic has system as separate parameter
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    conversation = [m for m in messages if m["role"] != "system"]
    
    return client.messages.create(
        model="claude-3-5-sonnet",
        system=system,
        messages=conversation,
        temperature=temperature
    )
```

## Debugging: Inspecting Messages

To see exactly what DSPy sends to your LLM:

```python
# After making a call
lm = dspy.settings.lm  # Your configured LM
last_call = lm.history[-1]

# View all messages
print("Messages sent to LLM:")
for i, msg in enumerate(last_call["messages"]):
    print(f"\n--- Message {i} ({msg['role']}) ---")
    print(msg["content"])
    print("-" * 50)

# View just system prompt
system_msg = last_call["messages"][0]
print("System prompt:")
print(system_msg["content"])

# Count demo messages
demo_count = (len(last_call["messages"]) - 2) // 2  # Exclude system and final user
print(f"\nNumber of demos: {demo_count}")
```

## Verification: Inspecting Messages in request_fn

You can verify this by printing messages in your request_fn:

```python
def ollama_request_fn(messages: list[dict[str, Any]], temperature: float = 0.0, **kwargs) -> dict:
    print(messages)  # Inspect what DSPy sends
    response = ollama_client.post(..., json={"messages": messages})
    return response.json()
```

**Key observation:** `messages` is already a complete, packed array with:
- `messages[0]` = System prompt (generated from signature)
- `messages[1..n-1]` = Demo pairs (if provided)
- `messages[n]` = Actual user input

You don't construct anything—just pass it through to your LLM provider.

### Real Example: Simple QA Bot

**Code:**
```python
from dspy.adapters.chat_adapter import ChatAdapter
import dspy

class QA(dspy.Signature):
    """Answer the question in concise."""
    question: str = dspy.InputField(desc="user's question")
    answer: str = dspy.OutputField(desc="answer to the question")

class QABot(dspy.Module):
    def __init__(self):
        self.program = dspy.Predict(QA)

    def forward(self, question):
        return self.program(question=question)

# Configure LM
dspy.configure(lm=your_lm)

# Create bot
qa_bot = QABot()

# Manually inspect what ChatAdapter generates
chat_adapter = ChatAdapter()
messages = chat_adapter.format(
    signature=QA,
    demos=[],
    inputs={"question": "what's love?"}
)

print(messages)
```

**Output (what gets sent to your LLM):**
```python
[
    {
        'role': 'system',
        'content': """Your input fields are:
1. `question` (str): user's question
Your output fields are:
1. `answer` (str): answer to the question
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Answer the question in concise."""
    },
    {
        'role': 'user',
        'content': """[[ ## question ## ]]
what's love?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."""
    }
]
```

**Key observations:**
1. **System message** contains:
   - Field descriptions (from `InputField`/`OutputField` desc)
   - Field structure template with `[[ ## field ## ]]` markers
   - Task objective (from signature docstring)

2. **User message** contains:
   - Actual input values formatted with field markers
   - Output format reminder

3. **No demos** = Only 2 messages (system + user)

4. **Your request_fn receives this exact array** - no construction needed!

## Summary

**The Flow:**
1. You provide: Field values + optional demos
2. DSPy generates: System prompt from signature
3. ChatAdapter constructs: Complete messages list
4. DriverLM receives: Fully formatted messages
5. Your request_fn: Adapts to provider API

**Your Responsibility:**
- Define signature (DSPy generates system prompt)
- Provide field values (DSPy formats into messages)
- Implement request_fn (adapt messages to provider)
- Implement output_fn (parse provider response)

**DSPy's Responsibility:**
- Generate system prompt from signature
- Format demos as message pairs
- Construct complete messages list
- Handle caching and history

**Mental Model:**
Think of `messages` in your `request_fn` as a **pre-packed array** where:
- System prompt is always `messages[0]`
- Everything is already formatted and ready
- You just adapt the format to your provider's API

This separation makes it easy to adapt DriverLM to any LLM provider—you just need to handle the API-specific request/response format, while DSPy handles all the prompt engineering.

## Quick Reference

**What you write:**
```python
class QA(dspy.Signature):
    """Answer questions."""
    question: str = dspy.InputField(desc="The question")
    answer: str = dspy.OutputField(desc="The answer")

qa = dspy.Predict(QA)
result = qa(question="What is love?")
```

**What your request_fn receives:**
```python
messages = [
    {"role": "system", "content": "Your input fields are:\n1. `question`...\n[[ ## question ## ]]\n{question}\n..."},
    {"role": "user", "content": "[[ ## question ## ]]\nWhat is love?\n\nRespond with..."}
]
```

**What you do:**
```python
def request_fn(messages, temperature=0.0, **kwargs):
    # Just pass to your provider
    return provider.chat(messages=messages, temperature=temperature)
```

That's it! DSPy handles everything else.
