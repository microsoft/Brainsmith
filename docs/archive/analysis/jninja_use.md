# Jinja Template Usage Reference

This document provides a synthesized reference for using the Jinja templating engine, based on the official documentation.

## 1. Introduction

Jinja is a modern and designer-friendly templating language for Python, modelled after Django’s templates. It is fast, widely used, and secure with optional sandboxed template execution.

- **Text-Based:** Generates any text-based format (HTML, XML, CSV, LaTeX, etc.).
- **No Specific Extension:** Templates are text files; `.html`, `.xml`, `.jinja`, etc., are all acceptable. A common practice is to store them in a `templates/` directory.
- **Syntax:** Inspired by Django and Python, featuring variables, expressions, and tags for logic.

## 2. Basic Syntax

### 2.1. Delimiters

Default delimiters are:
- `{% ... %}` for **Statements** (control flow like `if`, `for`).
- `{{ ... }}` for **Expressions** (variables or code to be printed to the output).
- `{# ... #}` for **Comments** (not included in the output).

These can be customized via the `Environment` settings.

### 2.2. Variables

- **Definition:** Variables are passed to the template via a context dictionary during rendering.
- **Access:** Use dot (`.`) or subscript (`[]`) notation.
  ```jinja
  {{ my_variable }}
  {{ my_dict.key }}
  {{ my_dict['key'] }}
  {{ my_object.attribute }}
  {{ my_list[0] }}
  ```
- **Undefined Variables:** Accessing a non-existent variable or attribute returns an `Undefined` object. By default, this evaluates to an empty string when printed or iterated over but raises an error for other operations. This behavior is configurable.
- **Python Methods:** You can call methods on objects passed to the template.
  ```jinja
  {{ my_string.capitalize() }}
  {{ "Hello, {}!".format(name) }}
  ```

### 2.3. Comments

- **Block Comments:** `{# This is a comment #}`. Can span multiple lines.
- **Line Comments:** If enabled via `line_comment_prefix` (e.g., `##`), they comment out the rest of the line.
  ```jinja
  ## This is a line comment
  ```

## 3. Control Structures (`{% ... %}`)

### 3.1. For Loops

Iterate over sequences (lists, tuples, dicts, etc.).

```jinja
<ul>
{% for user in users %}
  <li>{{ user.username|e }}</li>
{% else %}
  <li><em>No users found.</em></li>
{% endfor %}
</ul>
```

- **Loop Variable:** Inside a loop, the special `loop` variable provides context:
    - `loop.index`: Current iteration (1-indexed).
    - `loop.index0`: Current iteration (0-indexed).
    - `loop.revindex`: Iterations from the end (1-indexed).
    - `loop.revindex0`: Iterations from the end (0-indexed).
    - `loop.first`: True if first iteration.
    - `loop.last`: True if last iteration.
    - `loop.length`: Total number of items.
    - `loop.cycle(...)`: Cycle through values (e.g., `loop.cycle('odd', 'even')`).
    - `loop.previtem`: Item from the previous iteration (if exists).
    - `loop.nextitem`: Item from the next iteration (if exists).
    - `loop.changed(...)`: True if the value(s) changed from the previous iteration.
- **Filtering:** Skip items during iteration.
  ```jinja
  {% for user in users if not user.hidden %}
      ...
  {% endfor %}
  ```
- **Recursive Loops:** Use the `recursive` modifier and call `loop(new_iterable)`.
  ```jinja
  {% for item in sitemap recursive %}
      ...
      {% if item.children %}
          <ul>{{ loop(item.children) }}</ul>
      {% endif %}
  {% endfor %}
  ```
- **Loop Controls (Extension):** `break` and `continue` can be enabled via the `LoopControls` extension.

### 3.2. If Statements

Conditional logic, similar to Python.

```jinja
{% if user.is_authenticated %}
  Hello, {{ user.username }}!
{% elif user.is_anonymous %}
  Hello, Guest!
{% else %}
  Please log in.
{% endif %}
```

- **Tests:** Can be used directly in `if` conditions (e.g., `if variable is defined`).
- **Inline If:** `<expr1> if <condition> else <expr2>`
  ```jinja
  {{ 'Logged in' if user.is_authenticated else 'Guest' }}
  ```

### 3.3. Macros

Reusable template fragments, similar to functions.

```jinja
{% macro input(name, value='', type='text', size=20) -%}
    <input type="{{ type }}" name="{{ name }}" value="{{ value|e }}" size="{{ size }}">
{%- endmacro %}

{{ input('username') }}
{{ input('password', type='password') }}
```

- **Importing:** Macros can be imported from other files using `{% import %}` or `{% from ... import ... %}`.
  ```jinja
  {% import 'forms.html' as forms %}
  {{ forms.input('username') }}

  {% from 'forms.html' import input as input_field %}
  {{ input_field('password', type='password') }}
  ```
- **Context:** Imported templates don't get the current context by default (unless `with context` is used). Included templates do.
- **Scoping:** Macros defined in child templates do not override those in parent templates when called from the parent. Macros starting with `_` are private.

### 3.4. Call Blocks

Pass content into a macro, useful for wrapping blocks of markup.

```jinja
{% macro render_dialog(title, class='dialog') -%}
    <div class="{{ class }}">
        <h2>{{ title }}</h2>
        <div class="contents">
            {{ caller() }} {# Renders the content from the call block #}
        </div>
    </div>
{%- endmacro %}

{% call render_dialog('Hello World') %}
    This is the content of the dialog.
{% endcall %}
```
- **Arguments:** Caller can receive arguments: `{{ caller(user) }}` and `{% call(user) ... %}`.

### 3.5. Assignments (`set`)

Assign values to variables within the template scope.

```jinja
{% set navigation = [('index.html', 'Home'), ('about.html', 'About')] %}
{% set key = 'value' %}
```
- **Scope:** Assignments inside loops or blocks are local to that scope by default.
- **Namespace Object:** To modify variables across scopes (e.g., setting a flag inside a loop), use a `namespace` object.
  ```jinja
  {% set ns = namespace(found=false) %}
  {% for item in items %}
      {% if item.is_special %}
          {% set ns.found = true %}
      {% endif %}
  {% endfor %}
  Found special item: {{ ns.found }}
  ```
- **Block Assignments:** Capture rendered content into a variable. Filters can be applied.
  ```jinja
  {% set user_details | upper %}
  Name: {{ user.name }}
  Email: {{ user.email }}
  {% endset %}

  {{ user_details }}
  ```

### 3.6. Include

Include the rendered content of another template.

```jinja
{% include 'header.html' %}
Body
{% include 'footer.html' %}
```
- **Context:** Included templates receive the current context by default. Use `without context` to prevent this.
- **Ignoring Missing:** `{% include 'sidebar.html' ignore missing %}` prevents errors if the file doesn't exist.
- **List of Templates:** Try multiple templates in order: `{% include ['partial_user.html', 'partial_default.html'] %}`

### 3.7. Filters (Block)

Apply filters to a block of content.

```jinja
{% filter upper %}
    This text will be uppercase.
{% endfilter %}
```

### 3.8. Raw

Output content exactly as written, ignoring template syntax within the block.

```jinja
{% raw %}
    This will show {{ variable }} literally, not its value.
    {% if condition %}...{% endif %} will also be shown as text.
{% endraw %}
```

### 3.9. With Statement

Create a new inner scope, optionally assigning variables locally.

```jinja
{% with %}
    {% set inner_var = 'scoped' %}
    {{ inner_var }} {# Output: scoped #}
{% endwith %}
{# inner_var is not defined here #}

{% with outer_var=42, name='test' %}
    {{ outer_var }} {{ name }} {# Output: 42 test #}
{% endwith %}
```

## 4. Template Inheritance

Build a base "skeleton" template with common elements and define blocks that child templates can override.

### 4.1. Base Template (`base.html`)

```jinja
<!DOCTYPE html>
<html>
<head>
    {% block head %}
    <title>{% block title %}Default Title{% endblock %}</title>
    <link rel="stylesheet" href="style.css">
    {% endblock %}
</head>
<body>
    <div id="content">{% block content %}{% endblock %}</div>
    <div id="footer">
        {% block footer %}
        &copy; Copyright 2025
        {% endblock %}
    </div>
</body>
</html>
```

### 4.2. Child Template (`child.html`)

```jinja
{% extends "base.html" %} {# Must be the first tag #}

{% block title %}My Page Title{% endblock %}

{% block content %}
    <h1>Content Goes Here</h1>
    <p>This overrides the content block from base.html.</p>
{% endblock %}

{% block footer %}
    {{ super() }} {# Renders the content of the parent block #}
    - Powered by Jinja
{% endblock %}
```

- **`{% extends "parent.html" %}`:** Specifies the parent template. Must be the first tag in the file.
- **`{% block block_name %}`:** Defines a block that can be overridden by child templates.
- **`{{ super() }}`:** Renders the content of the block from the parent template. Can be chained (`super.super()`).
- **Named End Tags:** `{% endblock block_name %}` is allowed for clarity.
- **Block Scope:** Blocks don't access variables from outer scopes by default. Use `{% block block_name scoped %}` to allow access.
- **Required Blocks:** `{% block block_name required %}` forces child templates (at some level) to override the block.

## 5. Filters (`|`)

Modify variables. Applied using the pipe `|` operator. Can be chained.

```jinja
{{ name | striptags | upper | default('No Name') }}
```

### 5.1. Common Built-in Filters

- `abs`: Absolute value.
- `attr(attribute_name)`: Get an attribute (like `.attribute_name` but only checks attributes).
- `batch(linecount, fill_with=None)`: Batch items into lists of `linecount`.
- `capitalize`: Capitalize first letter, rest lowercase.
- `center(width=80)`: Center the string.
- `default(default_value='', boolean=False)` / `d`: Return default if value is undefined (or false if `boolean=True`).
- `dictsort`: Sort a dict by keys (or `value`), returns list of (key, value) tuples.
- `escape` / `e`: Escape HTML (`<`, `>`, `&`, `"`, `'`).
- `filesizeformat`: Human-readable file size.
- `first`: First item of a sequence.
- `float`: Convert to float (default 0.0).
- `forceescape`: Enforce escaping (can double-escape).
- `format`: C-style string formatting (`"%s %s"|format(a, b)`).
- `groupby(attribute)`: Group objects by an attribute.
- `indent(width=4, first=False, blank=False)`: Indent lines.
- `int`: Convert to integer (default 0).
- `join(separator='', attribute=None)`: Join a sequence with a separator.
- `last`: Last item of a sequence.
- `length` / `count`: Length of a sequence or mapping.
- `list`: Convert value to a list.
- `lower`: Convert to lowercase.
- `map(filter_name or attribute)`: Apply a filter or access an attribute on each item.
- `max`: Max item in a sequence.
- `min`: Min item in a sequence.
- `pprint`: Pretty print (for debugging).
- `random`: Random item from a sequence.
- `reject(test_name or attribute)`: Reject items where test is true.
- `rejectattr(attribute, test_name)`: Reject items where attribute passes test.
- `replace(old, new, count=None)`: Replace occurrences of a substring.
- `reverse`: Reverse a sequence or string.
- `round(precision=0, method='common'|'ceil'|'floor')`: Round a number.
- `safe`: Mark a string as safe (don't escape if autoescape is on).
- `select(test_name or attribute)`: Select items where test is true.
- `selectattr(attribute, test_name)`: Select items where attribute passes test.
- `slice(slices, fill_with=None)`: Slice an iterator into lists.
- `sort(reverse=False, case_sensitive=False, attribute=None)`: Sort an iterable.
- `string`: Convert object to string (preserves Markup safety).
- `striptags`: Remove SGML/XML tags.
- `sum(attribute=None, start=0)`: Sum of items in a sequence.
- `title`: Title-case the string.
- `tojson`: Convert object to JSON string (marked safe).
- `trim(chars=None)`: Strip leading/trailing characters (default whitespace).
- `truncate(length=255, killwords=False, end='...', leeway=None)`: Truncate string.
- `unique(case_sensitive=False, attribute=None)`: Unique items from an iterable.
- `upper`: Convert to uppercase.
- `urlencode`: Percent-encode for URLs.
- `urlize`: Convert URLs in text to clickable links.
- `wordcount`: Count words.
- `wordwrap`: Wrap text to a given width.
- `xmlattr`: Create SGML/XML attribute string from a dict.

### 5.2. Custom Filters

Define a Python function and register it with `environment.filters`.

```python
def my_reverse_filter(s):
    return s[::-1]

environment.filters['reverse_string'] = my_reverse_filter
```
```jinja
{{ "hello" | reverse_string }} {# Output: olleh #}
```
- Use `@pass_context`, `@pass_eval_context`, or `@pass_environment` decorators to get context/environment info passed to the filter.

## 6. Tests (`is`)

Check conditions on variables. Used with the `is` operator.

```jinja
{% if count is odd %} ... {% endif %}
{% if name is defined %} ... {% endif %}
```

### 6.1. Common Built-in Tests

- `boolean`: Is a boolean?
- `callable`: Is callable?
- `defined`: Is the variable defined?
- `divisibleby(num)`: Is divisible by `num`?
- `eq(other)` / `==`: Equal to `other`?
- `escaped`: Is the value already escaped Markup?
- `even`: Is an even number?
- `false`: Is the value `False`?
- `filter(name)`: Does a filter with `name` exist?
- `float`: Is a float?
- `ge(other)` / `>=`: Greater than or equal to `other`?
- `gt(other)` / `>`: Greater than `other`?
- `in(sequence)`: Is the value present in the sequence?
- `integer`: Is an integer?
- `iterable`: Is iterable?
- `le(other)` / `<=`: Less than or equal to `other`?
- `lower`: Is the string lowercased?
- `lt(other)` / `<`: Less than `other`?
- `mapping`: Is a mapping (dict)?
- `ne(other)` / `!=`: Not equal to `other`?
- `none`: Is the value `None`?
- `number`: Is a number (int or float)?
- `odd`: Is an odd number?
- `sameas(other)`: Is the exact same object (identity check)?
- `sequence`: Is a sequence (list, tuple, string)?
- `string`: Is a string?
- `test(name)`: Does a test with `name` exist?
- `true`: Is the value `True`?
- `undefined`: Is the variable undefined?
- `upper`: Is the string uppercased?

### 6.2. Custom Tests

Define a Python function and register it with `environment.tests`.

```python
def is_positive(n):
    return isinstance(n, (int, float)) and n > 0

environment.tests['positive'] = is_positive
```
```jinja
{% if value is positive %} ... {% endif %}
```
- Can also use context/environment decorators like filters.

## 7. Whitespace Control

- **Default:** Single trailing newline is stripped; other whitespace is preserved.
- **Configuration:**
    - `trim_blocks=True`: Removes the first newline after a block tag (`{% ... %}`).
    - `lstrip_blocks=True`: Strips leading spaces/tabs from the start of a line up to a block tag.
- **Manual Control:** Add a minus sign (`-`) to the start or end of any tag (`{%-`, `-%}`, `{{-`, `-}}`, `{#-`, `-#}`).
  ```jinja
  {% for item in seq -%} {{ item }} {%- endfor %} {# No whitespace between items #}
  ```
- **Line Statements:** If enabled (`line_statement_prefix`), they strip leading whitespace automatically.

## 8. Escaping

Preventing variable content from breaking markup (e.g., HTML).

### 8.1. Manual Escaping

- Use the `|e` or `|escape` filter.
  ```jinja
  {{ user_input | e }}
  ```
- Escape variables containing `<`, `>`, `&`, `"`, or `'` unless they are trusted, well-formed HTML.

### 8.2. Automatic Escaping

- Enabled via `Environment(autoescape=True)` or `Environment(autoescape=select_autoescape(...))`.
- `select_autoescape` enables/disables based on template file extension (e.g., enable for `.html`, `.xml`).
- All variables are escaped by default **unless** marked as safe.
- **Marking Safe:**
    - In Python: Wrap the string in `markupsafe.Markup`.
    - In Template: Use the `|safe` filter: `{{ trusted_html | safe }}`.
- **String Literals:** String literals in templates are considered **unsafe** by default when autoescaping is on.
- **Double Escaping:** Applying `|e` to an already escaped (but not marked safe) value will double-escape it. Applying `|e` to a `Markup` object does nothing.
- **Autoescape Block:** Temporarily override autoescaping within a template:
  ```jinja
  {% autoescape true %}
      Escaping is on here... {{ potentially_unsafe }}
  {% endautoescape %}

  {% autoescape false %}
      Escaping is off here... {{ pre_escaped_html }}
  {% endautoescape %}
  ```

## 9. Global Functions

Functions available in the global template scope.

- `range([start,] stop [, step])`: Generate a sequence of numbers (like Python's `range`, but returns a list/iterator depending on version/context).
- `lipsum(n=5, html=True, min=20, max=100)`: Generate Lorem Ipsum placeholder text.
- `dict(**items)`: Create a dictionary (e.g., `dict(key='value')`).
- `cycler(*items)`: Cycle through values across loops (use `.next()` and `.current`).
- `joiner(sep=', ')`: Helper to join sections, returning the separator except for the first call.
- `namespace(...)`: Create an object whose attributes can be set using `{% set ns.attr = value %}` to share state across scopes.

## 10. Python API Basics

### 10.1. Environment

The central object storing configuration, globals, filters, tests, and loaders.

```python
from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader("path/to/templates"),
    autoescape=select_autoescape(
        enabled_extensions=('html', 'xml'),
        default_for_string=True,
    ),
    trim_blocks=True,
    lstrip_blocks=True
)

# Add custom filters/tests/globals
env.filters['myfilter'] = my_filter_func
env.tests['mytest'] = my_test_func
env.globals['pi'] = 3.14159
```

### 10.2. Loaders

Responsible for finding and loading template source code.

- `FileSystemLoader(searchpath)`: Load from directories.
- `PackageLoader(package_name, package_path='templates')`: Load from a Python package.
- `DictLoader(mapping)`: Load from a Python dictionary.
- `FunctionLoader(load_func)`: Load using a custom function.
- `PrefixLoader(mapping)`: Delegates loading based on a template name prefix.
- `ChoiceLoader(loaders)`: Tries multiple loaders in order.
- `ModuleLoader(path)`: Loads pre-compiled templates.

### 10.3. Loading Templates

```python
# Load a specific template
template = env.get_template("my_template.html")

# Select the first available template from a list
template = env.select_template(["user_profile.html", "base_profile.html"])

# Load from a string
template = env.from_string("Hello {{ name }}!")
```

### 10.4. Rendering Templates

Pass context variables as keyword arguments or a dictionary.

```python
# Render to a string
output = template.render(name="World", items=[1, 2, 3])
# or
output = template.render({"name": "World", "items": [1, 2, 3]})

# Render piece by piece (generator)
for chunk in template.generate(name="World"):
    print(chunk)

# Render to a stream (for buffering control)
stream = template.stream(name="World")
stream.dump("output.html") # Write to file
```

### 10.5. Async Support

- Enable with `Environment(enable_async=True)`.
- Allows using `async def` functions and `await` within templates (implicitly).
- Requires an asyncio event loop.
- `template.render_async(...)` and `template.generate_async(...)` are used.

## 11. Extensions

Jinja can be extended with custom tags or features. Some built-in extensions (may require enabling in the `Environment`):

- **i18n:** Internationalization support (`{% trans %}`, `{% pluralize %}`, `_()`, `gettext()`).
- **Loop Controls:** Enables `break` and `continue` in loops.
- **With Statement:** (Enabled by default since 2.9) Provides `{% with %}` block for local scopes.
- **Autoescape:** (Enabled by default since 2.9) Provides `{% autoescape %}` block.
- **Debug:** Provides `{% debug %}` tag to dump context/filters/tests.

---
*This reference is based on Jinja documentation version ~3.x. Specific features and defaults might vary slightly between versions.*
