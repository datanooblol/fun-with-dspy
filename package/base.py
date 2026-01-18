"""
Generic DSPy BaseLM wrapper with pluggable drivers and output adapters.

This example includes:
- A reusable DriverLM(BaseLM)
- DSPy-compatible output normalization
- An Ollama backend implementation
"""

from typing import Callable, Any
from dspy.clients.base_lm import BaseLM
from datetime import datetime
import uuid as uuid_lib


# ----------------------------------------------------------------------
# 1. DSPy-Compatible Result Adapter
# ----------------------------------------------------------------------

class DSPyChoice:
    def __init__(self, text: str):
        self.message = type('Message', (), {'content': text, 'role': 'assistant'})()
        self.finish_reason = "stop"
        self.index = 0
    
    def __repr__(self):
        return f"Choices(finish_reason='{self.finish_reason}', index={self.index}, message=Message(content='{self.message.content}', role='{self.message.role}', tool_calls=None, function_call=None, provider_specific_fields=None, reasoning_content=None))"


class DSPyOutputItem:
    def __init__(self, text: str):
        self.type = "message"
        self.content = [type('obj', (), {'text': text})()]


class DSPyUsage(dict):
    """Usage statistics dict - populate manually in output_fn"""
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        super().__init__()
        self["prompt_tokens"] = prompt_tokens
        self["completion_tokens"] = completion_tokens
        self["total_tokens"] = total_tokens


class DSPyResult(dict):
    """
    ModelResponse-compatible object for DSPy.
    This is the CONTRACT for output_fn - it must return this type.
    
    Usage:
        def your_output_fn(raw_response: dict) -> DSPyResult:
            content = extract_content(raw_response)
            model = extract_model(raw_response)
            usage = DSPyUsage(
                prompt_tokens=raw_response.get('prompt_eval_count', 0),
                completion_tokens=raw_response.get('eval_count', 0),
                total_tokens=...
            )
            return DSPyResult(text=content, usage=usage, model=model)
    """
    def __init__(self, text: str, usage: DSPyUsage = None, model: str = "custom"):
        super().__init__()
        choice = DSPyChoice(text)
        self["choices"] = [choice]
        self.choices = [choice]
        self.output = [DSPyOutputItem(text)]
        self.usage = usage or DSPyUsage()
        self["usage"] = dict(self.usage)  # Store in dict for cache persistence
        self.model = model
        self.cache_hit = False
        
        # litellm-compatible attributes
        self.id = f"chatcmpl-{uuid_lib.uuid4()}"
        self.created = int(datetime.now().timestamp())
        self.object = "chat.completion"
        self.system_fingerprint = None
    
    def __repr__(self):
        return f"ModelResponse(id='{self.id}', created={self.created}, model='{self.model}', object='{self.object}', system_fingerprint={self.system_fingerprint}, choices={self.choices}, usage={dict(self.usage)}, cache_hit={self.cache_hit})"


# ----------------------------------------------------------------------
# 2. Generic Driver-Based BaseLM
# ----------------------------------------------------------------------

class DriverLM(BaseLM):
    """
    Generic DSPy LM that delegates:
    - request execution
    - output parsing

    to injected driver functions.
    
    Supports caching like dspy.LM for performance.
    """

    def __init__(
        self,
        request_fn: Callable[..., Any],
        output_fn: Callable[[Any], DSPyResult],
        temperature: float = 0.7,
        max_tokens: int = 256,
        cache: bool = True,
        **kwargs,
    ):
        super().__init__(model="custom", model_type="chat", cache=cache, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.request_fn = request_fn
        self.output_fn = output_fn
        self._setup_cached_forward()
    
    def _setup_cached_forward(self):
        """Setup cached forward function once during init"""
        from dspy.clients.cache import request_cache
        
        @request_cache(cache_arg_name="request", ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
        def cached_forward(request: dict):
            raw_result = self.request_fn(
                prompt=request.get("prompt"),
                messages=request.get("messages"),
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 256),
            )
            result = self.output_fn(raw_result)
            result.cache_hit = True
            return result
        
        self._cached_forward = cached_forward
    
    def _build_history_entry(self, prompt, messages, kwargs, result):
        """Build history entry following native dspy.LM contract"""
        # Get usage from dict (survives cache) or attribute (fresh result)
        usage_dict = result.get("usage", {}) or dict(result.usage) if hasattr(result, 'usage') else {}
        
        return {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": result,
            "outputs": [result.choices[0].message.content],
            "usage": usage_dict,
            "cost": 0.0,
            "timestamp": datetime.now().isoformat(),
            "uuid": str(uuid_lib.uuid4()),
            "model": result.model,
            "response_model": result.model,
            "model_type": self.model_type,
        }

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Allow calling the LM directly like native_lm('Hi')"""
        result = self.forward(prompt=prompt, messages=messages, **kwargs)
        # Return list of outputs like native dspy.LM
        return [result.choices[0].message.content]
    
    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)
        
        messages = messages or [{"role": "user", "content": prompt}]
        merged_kwargs = {**self.kwargs, **kwargs}
        
        request = dict(
            model="custom",
            messages=messages,
            **merged_kwargs
        )
        
        if cache:
            result = self._cached_forward(request=request)
        else:
            raw_result = self.request_fn(
                prompt=prompt,
                messages=messages,
                temperature=merged_kwargs.get("temperature", 0.7),
                max_tokens=merged_kwargs.get("max_tokens", 256),
            )
            result = self.output_fn(raw_result)
        
        # Log to history using contract
        history_kwargs = {k: v for k, v in merged_kwargs.items() if k not in ['temperature', 'max_tokens']}
        self.history.append(self._build_history_entry(prompt, messages, history_kwargs, result))
        
        return result

    async def aforward(self, *args, **kwargs):
        # Simple sync-to-async bridge
        return self.forward(*args, **kwargs)