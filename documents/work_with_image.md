# Working with Images in DSPy

## Overview

DSPy supports vision models through the `dspy.Image` type. This guide explains how images flow through DSPy and how to handle them in custom `request_fn` implementations.

---

## Creating Images

### Method 1: From File Path

```python
img = dspy.Image("path/to/image.jpg")
```

### Method 2: From Bytes

```python
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

img = dspy.Image(image_bytes)
```

### Method 3: From PIL Image

```python
from PIL import Image as PILImage

pil_img = PILImage.open("image.jpg")
img = dspy.Image(pil_img)
```

### Method 4: From URL

```python
img = dspy.Image("https://example.com/image.jpg")
```

---

## How Images Flow Through DSPy

### Step 1: User Creates Signature with Image Field

```python
class SceneDescription(dspy.Signature):
    """Describe the contents of an image in detail."""
    image: dspy.Image = dspy.InputField(desc="Image to describe")
    scene_description: str = dspy.OutputField(desc="Detailed description")
```

### Step 2: User Calls Predictor

```python
describe = dspy.Predict(SceneDescription)
img = dspy.Image("beach.jpg")
result = describe(image=img)
```

### Step 3: DSPy's ChatAdapter Formats the Image

**What DSPy does internally:**

```python
# Image.format() converts image to OpenAI-compatible format
formatted = img.format()
# Returns:
[{
    "type": "image_url",
    "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
    }
}]
```

### Step 4: ChatAdapter Builds Messages

**DSPy constructs messages in OpenAI format:**

```python
messages = [
    {
        "role": "system",
        "content": "Your input fields are:\n1. `image` (Image): Image to describe\n..."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "[[ ## image ## ]]\n<image>"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
                }
            },
            {"type": "text", "text": "\n\nRespond with the corresponding output fields..."}
        ]
    }
]
```

**Key insight:** The `content` field is now a **list** (multi-part content), not a string!

### Step 5: Your request_fn Receives These Messages

This is where you need to transform OpenAI format → Your LLM's format.

---

## Understanding request_fn for Images

### The Challenge

Different LLM providers expect images in different formats:

| Provider | Image Format |
|----------|-------------|
| OpenAI | `content: [{"type": "image_url", "image_url": {"url": "data:..."}}]` |
| Ollama | `images: ["base64string"]` (separate field) |
| AWS Bedrock | `content: [{"image": {"format": "png", "source": {"bytes": b"..."}}}]` |
| Anthropic | `content: [{"type": "image", "source": {"type": "base64", "data": "..."}}]` |

### The Pattern: Detect and Transform

**Universal pattern for handling images in request_fn:**

```python
def your_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    processed_messages = []
    
    for msg in messages:
        content = msg["content"]
        
        # KEY: Check if content is a list (multi-part)
        if isinstance(content, list):
            # Extract parts
            text_parts = [part["text"] for part in content if part.get("type") == "text"]
            image_parts = [part for part in content if part.get("type") == "image_url"]
            
            # Transform to your LLM's format
            processed_msg = transform_to_your_format(msg["role"], text_parts, image_parts)
            processed_messages.append(processed_msg)
        else:
            # Simple text message - pass through
            processed_messages.append(msg)
    
    # Call your LLM API
    return call_your_llm_api(processed_messages, temperature, max_tokens)
```

---

## Example 1: Ollama (Working Implementation)

### Ollama's Expected Format

```json
{
  "model": "llama3.2-vision:11b",
  "messages": [
    {
      "role": "user",
      "content": "Describe this image",
      "images": ["base64_string_without_header"]
    }
  ]
}
```

### Implementation

```python
import httpx

ollama_client = httpx.Client(timeout=600.0)

def ollama_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    processed_messages = []
    
    for msg in messages:
        content = msg["content"]
        
        # Handle multi-part content (text + images)
        if isinstance(content, list):
            # Extract text and images
            text_parts = [part["text"] for part in content if part.get("type") == "text"]
            image_parts = [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]
            
            processed_msg = {
                "role": msg["role"],
                "content": " ".join(text_parts)
            }
            
            # Ollama uses "images" field for base64 data
            if image_parts:
                processed_msg["images"] = [
                    # Extract base64 part (remove "data:image/jpeg;base64," prefix)
                    img.split(",")[1] if "base64," in img else img
                    for img in image_parts
                ]
            
            processed_messages.append(processed_msg)
        else:
            # Simple text message
            processed_messages.append(msg)
    
    response = ollama_client.post(
        'http://localhost:11434/api/chat',
        json={
            "model": "llama3.2-vision:11b",
            "messages": processed_messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    response.raise_for_status()
    return response.json()
```

### Why It Works

1. **Detects multi-part content**: `isinstance(content, list)`
2. **Separates text and images**: Different list comprehensions
3. **Transforms to Ollama format**: 
   - Text goes in `content` field
   - Images go in separate `images` field
   - Strips data URI prefix from base64
4. **Preserves text-only messages**: Pass through unchanged

---

## Example 2: AWS Bedrock

### Bedrock's Expected Format

```json
{
  "modelId": "us.amazon.nova-lite-v1:0",
  "messages": [
    {
      "role": "user",
      "content": [
        {"text": "Describe this image"},
        {
          "image": {
            "format": "jpeg",
            "source": {"bytes": b"..."}
          }
        }
      ]
    }
  ]
}
```

### Implementation

```python
import boto3
import base64

def bedrock_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=2048):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    system_messages = []
    conversation_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_messages.append({"text": msg["content"]})
        else:
            content = msg["content"]
            
            # Handle multi-part content
            if isinstance(content, list):
                bedrock_content = []
                
                for part in content:
                    if part.get("type") == "text":
                        bedrock_content.append({"text": part["text"]})
                    
                    elif part.get("type") == "image_url":
                        # Extract image data
                        image_url = part["image_url"]["url"]
                        
                        if image_url.startswith("data:"):
                            # Parse data URI: "data:image/jpeg;base64,..."
                            header, data = image_url.split(",", 1)
                            
                            # Extract format: "data:image/jpeg;base64" -> "jpeg"
                            format_type = header.split(";")[0].split("/")[1]
                            
                            # Decode base64 to bytes
                            image_bytes = base64.b64decode(data)
                            
                            bedrock_content.append({
                                "image": {
                                    "format": format_type,
                                    "source": {"bytes": image_bytes}
                                }
                            })
                
                conversation_messages.append({
                    "role": msg["role"],
                    "content": bedrock_content
                })
            else:
                # Simple text message
                conversation_messages.append({
                    "role": msg["role"],
                    "content": [{"text": content}]
                })
    
    request_params = {
        "modelId": "us.amazon.nova-lite-v1:0",
        "messages": conversation_messages,
        "inferenceConfig": {
            "temperature": temperature,
            "maxTokens": max_tokens,
        }
    }
    
    if system_messages:
        request_params["system"] = system_messages
    
    response = client.converse(**request_params)
    return response
```

### Why It Works

1. **Detects multi-part content**: `isinstance(content, list)`
2. **Iterates through parts**: Handles text and images separately
3. **Transforms to Bedrock format**:
   - Parses data URI to extract format and base64 data
   - Decodes base64 to bytes
   - Wraps in Bedrock's `{"image": {"format": "...", "source": {"bytes": ...}}}` structure
4. **Handles system messages**: Bedrock requires separate `system` parameter

---

## Thinking Framework for Any LLM

### Step 1: Understand Your LLM's API

**Questions to answer:**
1. How does it expect images? (separate field, inline, bytes, base64, URL?)
2. What format? (OpenAI-style, custom structure?)
3. Does it support multi-part content?
4. Any special requirements? (format field, MIME type, etc.)

### Step 2: Map OpenAI Format → Your Format

**OpenAI format (what DSPy gives you):**
```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
}
```

**Your LLM format (what you need to create):**
```python
# Example: Your custom format
{
    "role": "user",
    "message": "...",
    "attachments": [
        {"type": "image", "data": "...", "mime": "image/jpeg"}
    ]
}
```

### Step 3: Implement Transformation

```python
def your_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    transformed = []
    
    for msg in messages:
        content = msg["content"]
        
        if isinstance(content, list):
            # Multi-part: extract and transform
            text = extract_text(content)
            images = extract_images(content)
            
            transformed_msg = {
                "role": msg["role"],
                "message": text,
                "attachments": [
                    transform_image_to_your_format(img)
                    for img in images
                ]
            }
            transformed.append(transformed_msg)
        else:
            # Text-only: simple transform
            transformed.append({
                "role": msg["role"],
                "message": content
            })
    
    return call_your_api(transformed, temperature, max_tokens)
```

---

## Helper Functions

### Extract Text from Multi-Part Content

```python
def extract_text(content: list) -> str:
    """Extract all text parts from multi-part content."""
    text_parts = [part["text"] for part in content if part.get("type") == "text"]
    return " ".join(text_parts)
```

### Extract Images from Multi-Part Content

```python
def extract_images(content: list) -> list:
    """Extract all image URLs from multi-part content."""
    return [
        part["image_url"]["url"]
        for part in content
        if part.get("type") == "image_url"
    ]
```

### Parse Data URI

```python
def parse_data_uri(data_uri: str) -> tuple[str, bytes]:
    """
    Parse data URI into (format, bytes).
    
    Example: "data:image/jpeg;base64,/9j/4AAQ..." -> ("jpeg", b"...")
    """
    if not data_uri.startswith("data:"):
        raise ValueError("Not a data URI")
    
    # Split: "data:image/jpeg;base64,..." -> ["data:image/jpeg;base64", "..."]
    header, data = data_uri.split(",", 1)
    
    # Extract format: "data:image/jpeg;base64" -> "jpeg"
    mime_type = header.split(";")[0]  # "data:image/jpeg"
    format_type = mime_type.split("/")[1]  # "jpeg"
    
    # Decode base64
    import base64
    image_bytes = base64.b64decode(data)
    
    return format_type, image_bytes
```

---

## Complete Working Example

```python
import dspy
from package.base import DriverLM, ModelResponse, Usage
import httpx

# Setup Ollama client
ollama_client = httpx.Client(timeout=600.0)

def ollama_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    processed_messages = []
    for msg in messages:
        content = msg["content"]
        
        if isinstance(content, list):
            text_parts = [part["text"] for part in content if part.get("type") == "text"]
            image_parts = [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]
            
            processed_msg = {
                "role": msg["role"],
                "content": " ".join(text_parts)
            }
            
            if image_parts:
                processed_msg["images"] = [
                    img.split(",")[1] if "base64," in img else img
                    for img in image_parts
                ]
            
            processed_messages.append(processed_msg)
        else:
            processed_messages.append(msg)
    
    response = ollama_client.post(
        'http://localhost:11434/api/chat',
        json={
            "model": "llama3.2-vision:11b",
            "messages": processed_messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    response.raise_for_status()
    return response.json()

def ollama_output_fn(response: dict) -> ModelResponse:
    content = response.get("message", {}).get("content", "")
    model = response.get("model", "custom")
    
    usage = Usage(
        prompt_tokens=response.get("prompt_eval_count", 0),
        completion_tokens=response.get("eval_count", 0),
        total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
    )
    
    return ModelResponse.from_text(text=content.strip(), usage=usage, model=model)

# Create LM
lm = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=ollama_output_fn,
    cache=True
)
dspy.configure(lm=lm)

# Define signature
class SceneDescription(dspy.Signature):
    """Describe the contents of an image in detail."""
    image: dspy.Image = dspy.InputField(desc="Image to describe")
    scene_description: str = dspy.OutputField(desc="Detailed description")

# Use it
describe = dspy.Predict(SceneDescription)

# Method 1: From file path
img = dspy.Image("beach.jpg")
result = describe(image=img)
print(result.scene_description)

# Method 2: From bytes
with open("lake_mountain.jpg", "rb") as f:
    img = dspy.Image(f.read())
result = describe(image=img)
print(result.scene_description)
```

---

## Key Insights

1. **DSPy handles image encoding** - You receive base64 data URIs
2. **Multi-part content = list** - Check `isinstance(content, list)`
3. **Text-only content = string** - Pass through unchanged
4. **Each LLM is different** - Transform OpenAI format to your LLM's format
5. **Data URI structure** - `"data:image/jpeg;base64,<base64_data>"`
6. **Ollama wants base64 only** - Strip the prefix
7. **Bedrock wants bytes** - Decode base64 to bytes
8. **No changes to DriverLM needed** - All logic goes in `request_fn`

---

## Summary

**To support images in your custom LLM:**

1. **No changes to DriverLM** - It already works
2. **Update request_fn** - Handle multi-part content
3. **Check content type** - `isinstance(content, list)`
4. **Extract parts** - Separate text and images
5. **Transform format** - Map OpenAI → Your LLM
6. **Test with debug prints** - Verify transformations

The pattern is universal - only the transformation logic changes per LLM provider.
