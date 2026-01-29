# Understanding Custom Signature Implementation

## Overview

This document explains how to build a lightweight Signature system inspired by DSPy, focusing on understanding Python metaclasses and how they enable clean, declarative syntax.

## The Goal

We want to write this:

```python
class QA(Signature):
    """Answer questions accurately."""
    question = InputField(desc="The question to answer")
    answer = OutputField(desc="The answer")
```

And automatically get:
- `QA.input_fields` → `{'question': FieldInfo(...)}`
- `QA.output_fields` → `{'answer': FieldInfo(...)}`
- `QA.instructions` → `"Answer questions accurately."`

## Complete Implementation

```python
from dataclasses import dataclass
from typing import Dict, Type

@dataclass
class FieldInfo:
    """Container for field metadata."""
    name: str
    type: Type = str
    desc: str = ""
    prefix: str = ""
    is_input: bool = True
    
    def __post_init__(self):
        if not self.prefix:
            self.prefix = self._infer_prefix(self.name)
    
    @staticmethod
    def _infer_prefix(name: str) -> str:
        """Convert field_name -> Field Name"""
        return name.replace("_", " ").title()

def InputField(desc: str = "", prefix: str = "") -> FieldInfo:
    """Factory function for input fields."""
    return FieldInfo(name="", desc=desc, prefix=prefix, is_input=True)

def OutputField(desc: str = "", prefix: str = "") -> FieldInfo:
    """Factory function for output fields."""
    return FieldInfo(name="", desc=desc, prefix=prefix, is_input=False)

class SignatureMeta(type):
    """Metaclass that processes Signature class definitions."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        input_fields = {}
        output_fields = {}
        
        for field_name, field_value in list(namespace.items()):
            if isinstance(field_value, FieldInfo):
                field_value.name = field_name
                if field_value.is_input:
                    input_fields[field_name] = field_value
                else:
                    output_fields[field_name] = field_value
                del namespace[field_name]
        
        namespace['_input_fields'] = input_fields
        namespace['_output_fields'] = output_fields
        namespace['_instructions'] = (namespace.get('__doc__', '') or '').strip()
        
        return super().__new__(mcs, name, bases, namespace)
    
    @property
    def input_fields(cls):
        return cls._input_fields
    
    @property
    def output_fields(cls):
        return cls._output_fields
    
    @property
    def instructions(cls):
        return cls._instructions

class Signature(metaclass=SignatureMeta):
    """Base Signature class."""
    _input_fields = {}
    _output_fields = {}
    _instructions = ""
```

## Understanding Metaclasses

### What is a Metaclass?

A metaclass is a "class of a class" - it defines how classes behave. Just as a class defines how instances behave, a metaclass defines how classes behave.

```python
# Normal flow
class MyClass:        # type(MyClass) = type
    pass
obj = MyClass()       # type(obj) = MyClass

# With metaclass
class MyMeta(type):
    pass
class MyClass(metaclass=MyMeta):  # type(MyClass) = MyMeta
    pass
obj = MyClass()       # type(obj) = MyClass
```

### What is `namespace`?

**`namespace` is just a dictionary** that contains everything you define in a class.

#### Simple Example

```python
class MyClass:
    x = 10
    y = "hello"
    
    def method(self):
        pass
```

Python internally creates:
```python
namespace = {
    'x': 10,
    'y': "hello",
    'method': <function method>,
    '__module__': '__main__',
    '__qualname__': 'MyClass',
}
```

Then calls: `MyClass = type('MyClass', (), namespace)`

#### With Type Annotations

```python
class MyClass:
    x: int = 10
    y: str = "hello"
```

Namespace becomes:
```python
namespace = {
    'x': 10,
    'y': "hello",
    '__annotations__': {'x': int, 'y': str},  # Stored separately!
}
```

**CRITICAL**: Type annotations are stored in `__annotations__`, separate from the actual values.

#### In Our Signature Example

```python
class Calculator(Signature):
    """Solve math."""
    result: int = InputField(desc="The result")
```

**Step 1**: Python builds namespace:
```python
namespace = {
    '__doc__': "Solve math.",
    '__annotations__': {'result': int},           # Type annotation
    'result': FieldInfo(name="", type=str, ...),  # Field value (default type=str)
}
```

**Step 2**: Metaclass receives and modifies namespace:
```python
def __new__(mcs, name, bases, namespace, **kwargs):
    annotations = namespace.get('__annotations__', {})  # {'result': int}
    
    for field_name, field_value in namespace.items():
        if field_name == 'result':
            # Extract type from annotation
            field_value.type = annotations['result']  # Change str to int!
            
            # Move to _input_fields
            input_fields['result'] = field_value
            
            # Remove from namespace
            del namespace['result']
    
    # Add processed fields
    namespace['_input_fields'] = input_fields
```

**Step 3**: Final namespace becomes class attributes:
```python
Calculator.result        # AttributeError (deleted from namespace)
Calculator._input_fields # {'result': FieldInfo(type=int, ...)} (added to namespace)
```

### Visual Flow: Namespace Transformation

```
┌─────────────────────────────────────┐
│ namespace (before metaclass)        │
├─────────────────────────────────────┤
│ __doc__: "Solve math."              │
│ __annotations__: {'result': int}    │
│ result: FieldInfo(type=str, ...)    │
└─────────────────────────────────────┘
           ↓ Metaclass processes
┌─────────────────────────────────────┐
│ namespace (after metaclass)         │
├─────────────────────────────────────┤
│ __doc__: "Solve math."              │
│ _input_fields: {'result': ...}      │
│ _output_fields: {}                  │
│ _instructions: "Solve math."        │
│ (result deleted)                    │
└─────────────────────────────────────┘
           ↓ type.__new__()
┌─────────────────────────────────────┐
│ Calculator class                    │
├─────────────────────────────────────┤
│ Calculator._input_fields            │
│ Calculator._output_fields           │
│ Calculator._instructions            │
│ (Calculator.result doesn't exist)   │
└─────────────────────────────────────┘
```

### Why Manipulate Namespace?

**Problem**: We don't want `Calculator.result` to be a `FieldInfo` object.

**Solution**: 
1. Extract the `FieldInfo` from namespace
2. Store it in `_input_fields` dictionary
3. Delete the original attribute
4. Add `_input_fields` to namespace

**Result**: Clean API where fields are accessed via `Calculator.input_fields`, not `Calculator.result`.

### Key Namespace Operations

```python
def __new__(mcs, name, bases, namespace, **kwargs):
    # READ from namespace
    annotations = namespace.get('__annotations__', {})
    doc = namespace.get('__doc__', '')
    
    # ITERATE namespace
    for field_name, field_value in list(namespace.items()):
        if isinstance(field_value, FieldInfo):
            # MODIFY field
            field_value.name = field_name
            
            # DELETE from namespace
            del namespace[field_name]
    
    # ADD to namespace
    namespace['_input_fields'] = input_fields
    namespace['_output_fields'] = output_fields
    
    return super().__new__(mcs, name, bases, namespace)
```

### Critical Insights

1. **`namespace` is just a dict** - Contains all class attributes
2. **`__annotations__` is separate** - Type hints stored independently from values
3. **Metaclass can modify namespace** - Add, delete, or change any attribute
4. **Final namespace becomes the class** - Whatever's in namespace after `__new__` becomes class attributes
5. **This enables the "magic"** - Transform declarative syntax into processed data structures

### When Does a Metaclass Run?

**CRITICAL**: The metaclass runs **when the class is defined**, NOT when instances are created.

```python
class QA(Signature):  # ← SignatureMeta.__new__ runs HERE
    question = InputField()
    answer = OutputField()

# At this point, the class already has:
# - QA.input_fields
# - QA.output_fields
# - QA.instructions

# No instance needed!
print(QA.input_fields)  # Works immediately
```

### The `__new__` Method

`__new__` is called to create the class object itself. It receives:

```python
def __new__(mcs, name, bases, namespace, **kwargs):
    """
    mcs: The metaclass itself (SignatureMeta)
    name: Name of the class being created (e.g., "QA")
    bases: Tuple of parent classes (e.g., (Signature,))
    namespace: Dictionary of class attributes
    """
```

### Example: What Python Sees

When you write:

```python
class QA(Signature):
    """Answer questions."""
    question = InputField(desc="The question")
    answer = OutputField(desc="The answer")
```

Python internally does:

```python
# Step 1: Collect class attributes into namespace
namespace = {
    '__module__': '__main__',
    '__qualname__': 'QA',
    '__doc__': 'Answer questions.',
    'question': FieldInfo(name="", desc="The question", is_input=True),
    'answer': FieldInfo(name="", desc="The answer", is_input=False),
}

# Step 2: Call the metaclass
QA = SignatureMeta.__new__(
    SignatureMeta,  # mcs
    'QA',           # name
    (Signature,),   # bases
    namespace       # namespace
)
```

## Step-by-Step Walkthrough

### Step 1: Factory Functions Create Placeholders

```python
def InputField(desc: str = "", prefix: str = "") -> FieldInfo:
    return FieldInfo(name="", desc=desc, prefix=prefix, is_input=True)
    #              ^^^^^^^^ Empty! We don't know the field name yet
```

**Why empty name?** Because when you write `question = InputField()`, the function doesn't know it's being assigned to a variable called "question". Only the metaclass can see that.

### Step 2: Metaclass Collects Fields

```python
for field_name, field_value in list(namespace.items()):
    if isinstance(field_value, FieldInfo):
        # Found a field! Now we know its name
        field_value.name = field_name  # Fill in the blank!
        
        if field_value.is_input:
            input_fields[field_name] = field_value
        else:
            output_fields[field_name] = field_value
        
        # Remove from namespace - we don't want QA.question to be a FieldInfo
        del namespace[field_name]
```

**Key insight**: The metaclass sees `'question': FieldInfo(...)` in the namespace, so it knows:
1. The field name is `'question'`
2. The field value is the FieldInfo object
3. It can set `field_value.name = 'question'`

### Step 3: Store Processed Fields

```python
namespace['_input_fields'] = input_fields
namespace['_output_fields'] = output_fields
namespace['_instructions'] = (namespace.get('__doc__', '') or '').strip()
```

**Why underscore prefix?** Convention to indicate "private" attributes. Users access via properties.

### Step 4: Create the Class

```python
return super().__new__(mcs, name, bases, namespace)
```

This calls `type.__new__()` to actually create the class with our modified namespace.

### Step 5: Properties for Clean Access

```python
@property
def input_fields(cls):
    return cls._input_fields
```

**Why property?** So users can write `QA.input_fields` instead of `QA._input_fields`.

**Why `cls` not `self`?** Because this is accessed on the class itself, not an instance:
```python
QA.input_fields  # cls = QA (the class)
# NOT:
qa = QA()
qa.input_fields  # This would use self
```

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. You Write Code                                           │
│                                                             │
│   class QA(Signature):                                      │
│       question = InputField()                               │
│       answer = OutputField()                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Python Calls InputField() and OutputField()             │
│                                                             │
│   Returns: FieldInfo(name="", is_input=True)                │
│   Returns: FieldInfo(name="", is_input=False)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Python Builds Namespace Dictionary                      │
│                                                             │
│   namespace = {                                             │
│       'question': FieldInfo(name="", is_input=True),        │
│       'answer': FieldInfo(name="", is_input=False),         │
│   }                                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Python Calls SignatureMeta.__new__()                    │
│                                                             │
│   SignatureMeta.__new__(SignatureMeta, 'QA',               │
│                         (Signature,), namespace)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Metaclass Processes Fields                              │
│                                                             │
│   for field_name, field_value in namespace.items():        │
│       if isinstance(field_value, FieldInfo):               │
│           field_value.name = field_name  # "question"      │
│           input_fields['question'] = field_value           │
│           del namespace['question']                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Metaclass Modifies Namespace                            │
│                                                             │
│   namespace['_input_fields'] = {'question': ...}           │
│   namespace['_output_fields'] = {'answer': ...}            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Class is Created                                         │
│                                                             │
│   QA = type.__new__(SignatureMeta, 'QA',                   │
│                     (Signature,), modified_namespace)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. You Can Access Fields                                    │
│                                                             │
│   QA.input_fields   # {'question': FieldInfo(...)}         │
│   QA.output_fields  # {'answer': FieldInfo(...)}           │
└─────────────────────────────────────────────────────────────┘
```

## Common Misconceptions

### ❌ Misconception 1: "The metaclass runs when I create an instance"

```python
qa = QA()  # ← Metaclass does NOT run here
```

**Truth**: The metaclass runs when the class is **defined**, not when instances are created.

### ❌ Misconception 2: "I need to call something to process the fields"

```python
class QA(Signature):
    question = InputField()
    answer = OutputField()

# No need to do:
# QA.process_fields()  ← NOT NEEDED
# QA.initialize()      ← NOT NEEDED

# Fields are already processed!
print(QA.input_fields)  # Works immediately
```

### ❌ Misconception 3: "InputField knows the field name"

```python
question = InputField()  # InputField has NO IDEA it's called "question"
```

**Truth**: Only the metaclass can see the assignment `question = InputField()` and extract the name.

## Why This Design?

### Clean Syntax

Without metaclass:
```python
sig = Signature()
sig.add_input_field('question', desc="The question")
sig.add_output_field('answer', desc="The answer")
```

With metaclass:
```python
class QA(Signature):
    question = InputField(desc="The question")
    answer = OutputField(desc="The answer")
```

### Type Hints Work

```python
class QA(Signature):
    question: str = InputField()  # IDE knows question is a string
    answer: str = OutputField()
```

### Declarative Style

The class definition **declares** what the signature looks like, rather than imperatively building it.

## Comparison with DSPy

| Feature | DSPy | Our Implementation |
|---------|------|-------------------|
| Base | Pydantic BaseModel | Plain Python class |
| Metaclass | SignatureMeta(type(BaseModel)) | SignatureMeta(type) |
| Field validation | Pydantic validators | None (can add if needed) |
| Type coercion | Automatic | Manual |
| Complexity | High | Low |
| Dependencies | Pydantic | None |

## Key Takeaways

1. **Metaclasses run at class definition time**, not instance creation time
2. **`__new__` receives the class namespace** as a dictionary before the class exists
3. **Factory functions create placeholders** that the metaclass fills in
4. **Namespace manipulation** allows removing/adding class attributes
5. **Properties provide clean API** while hiding implementation details
6. **This pattern enables declarative syntax** similar to ORMs and web frameworks

## Type Checker Compatibility

### The Problem

When you write:
```python
class Calculator(Signature):
    result: int = InputField()
```

Type checkers complain:
```
Type "FieldInfo" is not assignable to declared type "int"
```

This happens because:
- You declared `result: int`
- But `InputField()` returns `FieldInfo`
- Type mismatch!

### Why DSPy Doesn't Have This Issue

DSPy (via Pydantic) uses a special return type:

```python
def InputField(desc: str = "", prefix: str = "") -> Any:  # Returns Any, not FieldInfo
    return FieldInfo(name="", desc=desc, prefix=prefix, is_input=True)
```

`Any` is compatible with any type annotation, so type checkers accept it.

### The Solution

**Change return type to `Any`:**

```python
from typing import Any

def InputField(desc: str = "", prefix: str = "") -> Any:
    return FieldInfo(name="", desc=desc, prefix=prefix, is_input=True)

def OutputField(desc: str = "", prefix: str = "") -> Any:
    return FieldInfo(name="", desc=desc, prefix=prefix, is_input=False)
```

### Trade-offs

| Approach | IDE Errors | Type Safety | Complexity |
|----------|-----------|-------------|------------|
| `-> Any` | ✅ None | ⚠️ Minimal | ✅ Simple |
| `-> FieldInfo` | ❌ Many | ✅ Full | ✅ Simple |
| Type stub `.pyi` | ✅ None | ✅ Full | ❌ Complex |
| `# type: ignore` | ✅ None | ❌ None | ⚠️ Per-line |

### What You Lose

Type checker won't know the return type:
```python
field = InputField()
field.name  # No autocomplete/type checking
```

**Solution**: Add explicit type hints where needed:
```python
def process_field(field: FieldInfo):  # Explicit type
    field.name  # Type checker knows now!
```

### What You Keep

1. **Runtime behavior unchanged**: `InputField()` still returns `FieldInfo`
2. **Type annotations work**: `field: int = InputField()` has no IDE errors
3. **Metaclass extracts types**: `Sig.input_fields['field'].type == int` still works

### Verdict

Using `-> Any` is the **standard approach** used by:
- Pydantic (millions of users)
- DSPy (production-ready)
- SQLAlchemy (ORM fields)

No future issues. The trade-off (less type safety in helper functions) is easily managed with explicit type hints where needed.

## Further Reading

- Python's data model: `type`, `__new__`, `__init__`
- Metaclass use cases: ORMs (SQLAlchemy), web frameworks (Django models)
- DSPy's actual implementation: `.venv/Lib/site-packages/dspy/signatures/signature.py`

## Working with Type Annotations

### The Problem

You want to write:

```python
class QA(Signature):
    """Answer questions."""
    question: str = InputField(desc="The question")
    answer: int = OutputField(desc="The answer as a number")
```

And have the type information (`str`, `int`) available for:
1. Generating better prompts for the LLM
2. Parsing/validating LLM responses
3. Type hints for IDEs

### Extracting Type Annotations

Python stores type hints in `__annotations__`:

```python
class QA(Signature):
    question: str = InputField()
    answer: int = OutputField()

# Python creates:
# __annotations__ = {'question': str, 'answer': int}
```

### Enhanced Metaclass Implementation

```python
class SignatureMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        input_fields = {}
        output_fields = {}
        
        # Get type annotations
        annotations = namespace.get('__annotations__', {})
        
        for field_name, field_value in list(namespace.items()):
            if isinstance(field_value, FieldInfo):
                field_value.name = field_name
                
                # Extract type from annotation if available
                if field_name in annotations:
                    field_value.type = annotations[field_name]
                
                if field_value.is_input:
                    input_fields[field_name] = field_value
                else:
                    output_fields[field_name] = field_value
                
                del namespace[field_name]
        
        namespace['_input_fields'] = input_fields
        namespace['_output_fields'] = output_fields
        namespace['_instructions'] = (namespace.get('__doc__', '') or '').strip()
        
        return super().__new__(mcs, name, bases, namespace)
```

### Using Types in Prompts

```python
def format_type_hint(field_type) -> str:
    """Convert Python type to LLM-friendly description."""
    type_map = {
        str: "a string",
        int: "an integer number",
        float: "a decimal number",
        bool: "true or false",
        list: "a list",
        dict: "a dictionary/object",
    }
    return type_map.get(field_type, "a value")

def format_prompt_with_types(signature, inputs_data):
    """Generate prompt with type information."""
    parts = []
    
    # Instructions
    if signature.instructions:
        parts.append(signature.instructions)
        parts.append("")
    
    # Input fields with type hints
    parts.append("Input:")
    for name, field in signature.input_fields.items():
        type_hint = format_type_hint(field.type)
        parts.append(f"- {name} ({type_hint}): {field.desc}")
    parts.append("")
    
    # Actual input values
    for name, field in signature.input_fields.items():
        if name in inputs_data:
            parts.append(f"<|{name}|>")
            parts.append(str(inputs_data[name]))
            parts.append("")
    
    # Output fields with type hints
    parts.append("Respond with:")
    for name, field in signature.output_fields.items():
        type_hint = format_type_hint(field.type)
        parts.append(f"<|{name}|> ({type_hint}): {field.desc}")
    
    return "\n".join(parts)
```

### Example Output

```python
class Calculator(Signature):
    """Perform calculations."""
    expression: str = InputField(desc="Math expression to evaluate")
    result: int = OutputField(desc="The calculated result")
    explanation: str = OutputField(desc="Step-by-step explanation")

# Generated prompt:
"""
Perform calculations.

Input:
- expression (a string): Math expression to evaluate

<|expression|>
2 + 2 * 3

Respond with:
<|result|> (an integer number): The calculated result
<|explanation|> (a string): Step-by-step explanation
"""
```

### Parsing with Type Coercion

```python
import json

def parse_value(value_str: str, expected_type):
    """Parse string value to expected type."""
    if expected_type == str:
        return value_str
    elif expected_type == int:
        return int(value_str.strip())
    elif expected_type == float:
        return float(value_str.strip())
    elif expected_type == bool:
        return value_str.strip().lower() in ('true', '1', 'yes')
    elif expected_type in (list, dict):
        return json.loads(value_str)
    else:
        return value_str

def parse_response(signature, response_text):
    """Parse LLM response with type coercion."""
    parser = CompactFieldParser()
    raw_fields = parser.parse(response_text, list(signature.output_fields.keys()))
    
    # Convert to correct types
    typed_fields = {}
    for name, value_str in raw_fields.items():
        field = signature.output_fields[name]
        try:
            typed_fields[name] = parse_value(value_str, field.type)
        except Exception as e:
            raise ValueError(f"Failed to parse {name} as {field.type}: {e}")
    
    return typed_fields
```

### Working with Complex Types

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Person:
    name: str
    age: int

class UserQuery(Signature):
    """Extract user information."""
    text: str = InputField(desc="Raw text containing user info")
    users: List[Person] = OutputField(desc="List of extracted users")

# In prompt:
"""
Respond with:
<|users|> (a list): List of extracted users
Format as JSON: [{"name": "...", "age": ...}, ...]
"""

# Parsing:
def parse_dataclass_list(value_str: str, item_type):
    data = json.loads(value_str)
    return [item_type(**item) for item in data]
```

### Complete Example

```python
class MathProblem(Signature):
    """Solve math problems step by step."""
    problem: str = InputField(desc="The math problem to solve")
    steps: List[str] = OutputField(desc="List of solution steps")
    answer: float = OutputField(desc="Final numerical answer")

# Usage:
response = """
<|steps|>
["First, multiply 2 * 3 = 6", "Then, add 2 + 6 = 8"]

<|answer|>
8.0
"""

result = parse_response(MathProblem, response)
# result = {
#     'steps': ["First, multiply 2 * 3 = 6", "Then, add 2 + 6 = 8"],
#     'answer': 8.0
# }
```

## Key Insights

1. **Type annotations are stored in `__annotations__`** - The metaclass can access them
2. **Types improve prompts** - Tell the LLM what format to use
3. **Types enable parsing** - Convert string responses to correct Python types
4. **Complex types need JSON** - Use JSON for lists, dicts, and dataclasses
5. **Type hints help IDEs** - Better autocomplete and type checking

## Exercise

Try modifying the metaclass to:
1. Automatically infer field types from type hints ✅ (shown above)
2. Validate that all InputFields come before OutputFields
3. Add a `required` parameter to fields and validate it
4. Support Union types (e.g., `str | int`)
5. Generate JSON schema from type annotations

This will deepen your understanding of how metaclasses work!
