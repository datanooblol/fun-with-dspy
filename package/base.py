"""
Generic DSPy BaseLM wrapper with pluggable drivers and output adapters.

This example includes:
- A reusable DriverLM(BaseLM)
- DSPy-compatible output normalization
- An Ollama backend implementation
"""

from typing import Callable, Any, Optional, List
from dspy.clients.base_lm import BaseLM
from datetime import datetime
import uuid as uuid_lib
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# 1. DSPy-Compatible Result Adapter (Pydantic Models)
# ----------------------------------------------------------------------

class Message(BaseModel):
    content: str
    role: str = "assistant"

class Choice(BaseModel):
    finish_reason: str = "stop"
    index: int = 0
    message: Message
    
    def __repr__(self):
        content_repr = repr(self.message.content)  # Escapes \n
        return f"Choices(finish_reason='{self.finish_reason}', index={self.index}, message=Message(content={content_repr}, role='{self.message.role}', tool_calls=None, function_call=None, provider_specific_fields=None, reasoning_content=None))"

class Usage(BaseModel):
    """Usage statistics - populate in output_fn"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ModelResponse(BaseModel):
    """
    ModelResponse-compatible object for DSPy.
    This is the CONTRACT for output_fn - it must return this type.
    
    Usage:
        def your_output_fn(raw_response: dict) -> ModelResponse:
            content = extract_content(raw_response)
            model = extract_model(raw_response)
            usage = Usage(
                prompt_tokens=raw_response.get('prompt_eval_count', 0),
                completion_tokens=raw_response.get('eval_count', 0),
                total_tokens=...
            )
            return ModelResponse.from_text(text=content, usage=usage, model=model)
    """
    model_config = {"arbitrary_types_allowed": True}
    
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid_lib.uuid4()}")
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str = "custom"
    object: str = "chat.completion"
    system_fingerprint: Optional[str] = None
    choices: List[Choice]
    usage: dict = Field(default_factory=dict)  # Store as dict for cache compatibility
    cache_hit: bool = False
    
    @classmethod
    def from_text(cls, text: str, usage: Usage = None, model: str = "custom"):
        """Convenience constructor from text"""
        message = Message(content=text, role="assistant")
        choice = Choice(message=message)
        usage_dict = usage.model_dump() if usage else {}
        return cls(
            choices=[choice],
            usage=usage_dict,
            model=model
        )
    
    def __repr__(self):
        content = self.choices[0].message.content if self.choices else ""
        content_repr = repr(content)
        return f"ModelResponse(id='{self.id}', created={self.created}, model='{self.model}', object='{self.object}', system_fingerprint={self.system_fingerprint}, choices=[Choices(finish_reason='{self.choices[0].finish_reason}', index={self.choices[0].index}, message=Message(content={content_repr}, role='{self.choices[0].message.role}', tool_calls=None, function_call=None, provider_specific_fields=None, reasoning_content=None))], usage={self.usage}, cache_hit={self.cache_hit})"

# Backwards compatibility
DSPyResult = ModelResponse
DSPyUsage = Usage


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
    
    def clear_cache(self):
        """Clear the request cache and recreate cached function"""
        if hasattr(self._cached_forward, 'cache_clear'):
            self._cached_forward.cache_clear()
        # Recreate the cached function to ensure clean state
        self._setup_cached_forward()
    
    def _setup_cached_forward(self):
        """Setup cached forward function once during init"""
        from dspy.clients.cache import request_cache
        import functools
        
        # Track if function actually executed
        self._last_call_was_cached = False
        
        def uncached_forward(request: dict):
            self._last_call_was_cached = False  # Mark as fresh call
            raw_result = self.request_fn(
                prompt=request.get("prompt"),
                messages=request.get("messages"),
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 256),
            )
            result = self.output_fn(raw_result)
            return result
        
        # Apply cache decorator
        cached_fn = request_cache(cache_arg_name="request", ignored_args_for_cache_key=["api_key", "api_base", "base_url"])(uncached_forward)
        
        # Store both versions
        self._uncached_forward = uncached_forward
        self._cached_forward = cached_fn
    
    def _build_history_entry(self, prompt, messages, kwargs, result):
        """Build history entry following native dspy.LM contract"""
        return {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": result,
            "outputs": [result.choices[0].message.content],
            "usage": result.usage,
            "cost": 0.0,
            "timestamp": datetime.now().isoformat(),
            "uuid": str(uuid_lib.uuid4()),
            "model": result.model,
            "response_model": result.model,
            "model_type": self.model_type,
        }

    def __call__(self, prompt:str|None=None, messages:list[dict[str,Any]]|None=None, **kwargs)->List[dict[str, Any] | str]:
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
            self._last_call_was_cached = True  # Assume cached
            result = self._cached_forward(request=request)
            # If function executed, _last_call_was_cached is now False
            result.cache_hit = self._last_call_was_cached
        else:
            raw_result = self.request_fn(
                prompt=prompt,
                messages=messages,
                temperature=merged_kwargs.get("temperature", 0.7),
                max_tokens=merged_kwargs.get("max_tokens", 256),
            )
            result = self.output_fn(raw_result)
            result.cache_hit = False
        
        # Log to history using contract
        history_kwargs = {k: v for k, v in merged_kwargs.items() if k not in ['temperature', 'max_tokens']}
        self.history.append(self._build_history_entry(prompt, messages, history_kwargs, result))
        
        return result

    async def aforward(self, *args, **kwargs):
        # Simple sync-to-async bridge
        return self.forward(*args, **kwargs)