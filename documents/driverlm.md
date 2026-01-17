# DriverLM: Custom LM Integration for DSPy

## Why DriverLM?

**Problem**: DSPy's native `dspy.LM` works great with catalog models (OpenAI, Anthropic, etc.) but doesn't reflect real-world production scenarios where you need to:
- Use custom models not in the LiteLLM catalog
- Optimize prompts with the **exact model** you'll deploy to production
- Maintain control over model serving infrastructure
- Ensure prompt optimization results are production-ready

**Solution**: DriverLM allows you to plug any custom LLM backend into DSPy while maintaining full compatibility with DSPy's optimization tools (GEPA, BootstrapFewShot, etc.).

## Key Insight

> **Optimize with what you'll deploy.** If you optimize prompts with GPT-4 but deploy with a custom Llama model, your optimized prompts won't work as expected. DriverLM ensures your optimization happens with your production model.

## Implementation

### 1. Core DriverLM Class

```python
from typing import Callable, Any
from dspy.clients.base_lm import BaseLM

class DSPyChoice:
    def __init__(self, text: str):
        self.message = type('obj', (), {'content': text})()
        self.finish_reason = "stop"

class DSPyUsage(dict):
    def __init__(self):
        super().__init__()
        self["prompt_tokens"] = 0
        self["completion_tokens"] = 0
        self["total_tokens"] = 0

class DSPyResult(dict):
    """OpenAI-compatible response format for DSPy"""
    def __init__(self, text: str):
        super().__init__()
        choice = DSPyChoice(text)
        self["choices"] = [choice]
        self.choices = [choice]
        self.usage = DSPyUsage()
        self.model = "custom"
        self.cache_hit = False

class DriverLM(BaseLM):
    def __init__(
        self,
        request_fn: Callable[..., Any],
        output_fn: Callable[[Any], DSPyResult],
        temperature: float = 0.7,
        max_tokens: int = 256,
        cache: bool = True,
        **kwargs,
    ):
        super().__init__(model="custom", model_type="chat", cache=cache, 
                         temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.request_fn = request_fn
        self.output_fn = output_fn

    def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **kwargs):
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)
        
        messages = messages or [{"role": "user", "content": prompt}]
        merged_kwargs = {**self.kwargs, **kwargs}
        
        request = dict(model="custom", messages=messages, **merged_kwargs)
        
        if cache:
            from dspy.clients.cache import request_cache
            
            @request_cache(cache_arg_name="request", 
                          ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
            def cached_forward(request: dict):
                raw_result = self.request_fn(
                    prompt=request.get("prompt"),
                    messages=request.get("messages"),
                    temperature=request.get("temperature", 0.7),
                    max_tokens=request.get("max_tokens", 256),
                )
                return self.output_fn(raw_result)
            
            return cached_forward(request=request)
        else:
            raw_result = self.request_fn(
                prompt=prompt,
                messages=messages,
                temperature=merged_kwargs.get("temperature", 0.7),
                max_tokens=merged_kwargs.get("max_tokens", 256),
            )
            return self.output_fn(raw_result)

    async def aforward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

### 2. Ollama Integration Example

```python
import httpx
from package.base import DSPyResult, DriverLM

# Global httpx client for connection pooling (same as LiteLLM)
ollama_client = httpx.Client(timeout=600.0)

def ollama_request_fn(prompt=None, messages=None, temperature=0.0, max_tokens=256):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    response = ollama_client.post(
        'http://localhost:11434/api/chat',
        json={
            "model": "llama3.2-vision:11b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    response.raise_for_status()
    return {"response": response.json()["message"]["content"]}

def ollama_output_fn(response):
    return DSPyResult(response.get("response", "").strip())

# Create DriverLM instance
lm = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=ollama_output_fn,
    temperature=0.0
)

# Use with DSPy
import dspy
dspy.configure(lm=lm)
```

## Critical Implementation Details

### 1. Caching is Essential

**Problem Encountered**: Initial DriverLM implementation was 10x slower than `dspy.LM` during GEPA optimization.

**Root Cause**: DSPy's `dspy.LM` has caching enabled by default (`cache=True`). During optimization:
- GEPA makes 100+ LLM calls
- Many calls are identical (same messages, same temperature)
- Without caching, DriverLM made every single call
- With caching, `dspy.LM` skipped duplicate calls

**Evidence**: 
- `dspy.LM` jumped from iteration 1→78→126 (cache hits)
- DriverLM incremented by 3 each time (no cache, making all calls)

**Solution**: Implement DSPy's `request_cache` decorator to cache the final `DSPyResult` object.

**Key Learning**: Cache the **final output** (`DSPyResult`), not the raw response. This ensures cache returns the correct format DSPy expects.

### 2. Connection Pooling

**Problem**: Creating new HTTP connections for every request adds 50-100ms overhead per call.

**Solution**: Use `httpx.Client` (same as LiteLLM) instead of `requests.post()`:
```python
# Bad: Creates new connection every time
response = requests.post(url, json=data)

# Good: Reuses connections
ollama_client = httpx.Client(timeout=600.0)
response = ollama_client.post(url, json=data)
```

**Why httpx?**
- Built-in connection pooling
- HTTP/2 support (more efficient than HTTP/1.1)
- Same library LiteLLM uses internally

### 3. Score Alignment

**Problem**: Native `dspy.LM` achieved higher scores than DriverLM in GEPA optimization.

**Root Causes**:
1. **Cache pollution**: If cache wasn't cleared between tests, the second test benefited from cached results
2. **Stochastic optimization**: GEPA uses `temperature=1.0` for reflection, so different runs explore different paths

**Solution**: 
```python
# Clear cache between tests
if hasattr(dspy, 'cache'):
    dspy.cache.reset_memory_cache()
```

**Key Learning**: Score differences in optimization are often due to randomness, not implementation bugs. Verify basic functionality first (same input → same output) before debugging optimization scores.

## Performance Comparison

After implementing caching and connection pooling:

| Metric | dspy.LM | DriverLM | Status |
|--------|---------|----------|--------|
| Basic inference | ✓ | ✓ | ✅ Identical |
| Caching behavior | 1→78→126 jumps | 1→78→126 jumps | ✅ Identical |
| GEPA optimization speed | Fast | Fast | ✅ Identical |
| GEPA final scores | 0.83 | 0.83 | ✅ Identical |

## Usage with DSPy Optimizers

DriverLM works with all DSPy optimizers:

```python
from dspy.teleprompt import GEPA, BootstrapFewShot

# GEPA optimization
optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    reflection_lm=dspy.LM(model="gpt-4", temperature=1.0)  # Can use any LM for reflection
)
optimized = optimizer.compile(student, trainset=train, valset=val)

# BootstrapFewShot
optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized = optimizer.compile(student, trainset=train)
```

## Key Takeaways

1. **Caching is not optional** - It's essential for performance parity with `dspy.LM`
2. **Use httpx, not requests** - Connection pooling matters for 100+ calls
3. **Cache the final output** - Cache `DSPyResult`, not raw responses
4. **Clear cache between tests** - Avoid cache pollution in comparisons
5. **Optimize with production models** - Don't optimize with GPT-4 if you deploy with Llama
6. **Score variance is normal** - Stochastic optimizers produce different results each run

## Real-World Benefits

1. **Production alignment**: Prompts optimized with your actual deployment model
2. **Cost control**: Use your own infrastructure instead of API calls
3. **Model flexibility**: Swap models without changing DSPy code
4. **Easier maintenance**: Optimized prompts work with the model you tested
5. **Full control**: Custom preprocessing, postprocessing, or model serving logic

## Example: Complete GEPA Workflow

```python
import dspy
import httpx
from dspy.teleprompt import GEPA

# Setup DriverLM with Ollama
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

lm = DriverLM(
    request_fn=ollama_request_fn,
    output_fn=lambda r: DSPyResult(r.get("response", "").strip()),
    temperature=0.0
)

dspy.configure(lm=lm)

# Define task
class EmotionClassifier(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought("sentence -> emotion")
    
    def forward(self, sentence):
        return self.predict(sentence=sentence)

# Optimize with GEPA
optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    reflection_lm=dspy.LM(model="ollama/llama3.2-vision:11b", 
                          api_base="http://localhost:11434", temperature=1.0)
)

optimized = optimizer.compile(EmotionClassifier(), trainset=train, valset=val)
optimized.save("emotion_classifier_optimized.json")
```

The optimized prompts are now guaranteed to work with your production Llama model, not just in theory with GPT-4.
