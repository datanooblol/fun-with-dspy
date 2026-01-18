# DSPy Cache Behavior

## Overview
DSPy implements a 2-level caching system:
1. **In-memory cache** - LRUCache (fast, session-scoped)
2. **Disk cache** - FanoutCache (persistent across sessions)

## Key Behaviors

### Cache Persistence
- Disk cache persists across Python sessions/kernel restarts
- Cache is stored globally, shared across all LM instances in the same process
- Cache location: `.venv/Lib/site-packages/dspy/clients/cache.py` (uses `diskcache.FanoutCache`)

### Usage Clearing on Cache Hits
**IMPORTANT**: DSPy intentionally clears `usage` data on cache hits:
```python
# From dspy/clients/cache.py line 127-130
if hasattr(response, "usage"):
    # Clear the usage data when cache is hit, because no LM call is made
    response.usage = {}
    response.cache_hit = True
```

This is by design - when a response comes from cache, no actual LM call was made, so usage is empty.

### Expected Behavior
- **First call** (cache miss): `cache_hit=False`, `usage={'prompt_tokens': X, 'completion_tokens': Y, ...}`
- **Second call** (cache hit): `cache_hit=True`, `usage={}`

## Clearing Cache

### Method 1: Clear Disk Cache (Recommended)
```python
import dspy

# Clear global disk cache before creating LM
if hasattr(dspy, 'cache') and hasattr(dspy.cache, 'disk_cache'):
    dspy.cache.disk_cache.clear()
```

### Method 2: Clear Instance Cache (In-memory only)
```python
native_lm = DriverLM(...)
native_lm.clear_cache()  # Only clears in-memory cache for this instance
```

### Method 3: Disable Cache
```python
native_lm = DriverLM(..., cache=False)  # No caching at all
```

## Common Issues

### Issue: Old cache persists after kernel restart
**Cause**: Disk cache persists across sessions
**Solution**: Clear disk cache with `dspy.cache.disk_cache.clear()`

### Issue: Usage is empty even on first call
**Cause**: Hitting old disk cache from previous session
**Solution**: Clear disk cache before first call

### Issue: Changed code structure but getting old cached responses
**Cause**: Cache key is based on request parameters, not code structure
**Solution**: Clear disk cache or change request parameters

## Best Practices

1. **Development**: Clear disk cache at notebook start
   ```python
   if hasattr(dspy, 'cache') and hasattr(dspy.cache, 'disk_cache'):
       dspy.cache.disk_cache.clear()
   ```

2. **Production**: Keep cache enabled for performance
   ```python
   native_lm = DriverLM(..., cache=True)
   ```

3. **Testing**: Disable cache to ensure fresh calls
   ```python
   native_lm = DriverLM(..., cache=False)
   ```

## Implementation Notes

### DriverLM Cache Integration
- Uses `@request_cache` decorator from `dspy.clients.cache`
- Cache key computed from request dict (messages, temperature, etc.)
- `cache_hit` flag set dynamically after cache lookup
- `_last_call_was_cached` flag tracks whether cached function executed

### ModelResponse Structure
- `usage` field stored as `dict` (not Pydantic `Usage` object) for cache compatibility
- `cache_hit` field set to `False` by default, updated to `True` on cache hits
- Pydantic serialization ensures proper cache storage/retrieval
