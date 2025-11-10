# API Documentation Template

This template defines the standard format for API reference pages.

## File Structure

```markdown
# Module Name

Brief one-line description of the module.

More detailed explanation if needed (1-2 sentences).

---

::: module.path.function_name

**Example:**

```python
from module import function_name

# Minimal, focused example
result = function_name(arg1="value", arg2=42)
print(result)
```

---

::: module.path.another_function

**Example:**

```python
# Another focused example
```

---

## See Also

- [External Link](url) - Description
- [Another Resource](url) - Description
```

## Rules

### ✅ DO

- **Keep it flat** - No manual section headings (mkdocstrings generates them)
- **One example per entry** - Focused, runnable code showing typical usage
- **Use horizontal rules** - Separate entries with `---` for clarity
- **Brief module intro** - 1-2 sentences at the top explaining the module's purpose
- **End with "See Also"** - Links to external resources (other docs, GitHub, etc.)

### ❌ DON'T

- **No admonitions** - No tips, warnings, or info boxes (save for tutorials)
- **No section groupings** - No "High-Level API" or "Helper Functions" sections
- **No advice/guidance** - Just facts and examples (guidance goes in tutorials)
- **No "Related:" links** - The table of contents provides navigation
- **No multiple examples** - Keep one focused example per function/class

## Example Entry

**Good:**

```markdown
---

::: brainsmith.dse.explore_design_space

**Example:**

```python
from brainsmith import explore_design_space

results = explore_design_space(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml"
)

stats = results.compute_stats()
print(f"Success: {stats['successful']}")
```

---
```

**Bad:**

```markdown
---

## High-Level API

### explore_design_space

!!! tip "Most users should use this function"
    This is the recommended entry point for DSE.

::: brainsmith.dse.explore_design_space

**Basic Example:**

```python
# Basic usage
results = explore_design_space(...)
```

**Advanced Example:**

```python
# With all options
results = explore_design_space(..., verbose=True, ...)
```

**Related:** [TreeExecutionResult](#treeexecutionresult)

---
```

## Why This Format?

1. **Clean TOC** - No duplicate headings from manual sections
2. **Scannable** - Consistent pattern makes it easy to find what you need
3. **Reference-focused** - Facts and examples, not explanations
4. **Maintainable** - Simple template that's easy to follow

## Global Settings

Configure mkdocstrings options globally in `mkdocs.yml`:

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_signature_annotations: true
            separate_signature: true
            # ... other options
```

This eliminates repetition in individual pages.
