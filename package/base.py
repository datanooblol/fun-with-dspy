"""
Generic DSPy BaseLM wrapper with pluggable drivers and output adapters.

This example includes:
- A reusable DriverLM(BaseLM)
- DSPy-compatible output normalization
- An Ollama backend implementation
"""

from typing import Callable, Any
from dspy.clients.base_lm import BaseLM


# ----------------------------------------------------------------------
# 1. DSPy-Compatible Result Adapter
# ----------------------------------------------------------------------

class DSPyChoice:
    def __init__(self, text: str):
        self.message = type('obj', (), {'content': text})()
        self.finish_reason = "stop"


class DSPyOutputItem:
    def __init__(self, text: str):
        self.type = "message"
        self.content = [type('obj', (), {'text': text})()]


class DSPyUsage(dict):
    def __init__(self):
        super().__init__()
        self["prompt_tokens"] = 0
        self["completion_tokens"] = 0
        self["total_tokens"] = 0


class DSPyResult(dict):
    """
    Minimal OpenAI-like response object that DSPy expects.
    """
    def __init__(self, text: str):
        super().__init__()
        choice = DSPyChoice(text)
        self["choices"] = [choice]
        self.choices = [choice]
        self.output = [DSPyOutputItem(text)]
        self.usage = DSPyUsage()
        self.model = "custom"
        self.cache_hit = False


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
            return self.output_fn(raw_result)
        
        self._cached_forward = cached_forward

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
            return self._cached_forward(request=request)
        else:
            raw_result = self.request_fn(
                prompt=prompt,
                messages=messages,
                temperature=merged_kwargs.get("temperature", 0.7),
                max_tokens=merged_kwargs.get("max_tokens", 256),
            )
            return self.output_fn(raw_result)

    async def aforward(self, *args, **kwargs):
        # Simple sync-to-async bridge
        return self.forward(*args, **kwargs)