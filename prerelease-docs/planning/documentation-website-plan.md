# Brainsmith Documentation Website Plan

## Executive Summary

**Recommendation: Material for MkDocs + mkdocstrings**

This is the modern industry standard for Python projects, used by **FastAPI, Pydantic, Typer, Microsoft, Google, NVIDIA, and many top open-source projects**.

## Core Technology Stack

### 1. Material for MkDocs (Base framework)
- Beautiful, modern UI that looks professional out-of-the-box
- Built-in search with offline support
- Mobile-responsive design
- Native dark/light mode
- Live preview during development (`mkdocs serve`)
- One-command GitHub Pages deployment (`mkdocs gh-deploy`)

### 2. mkdocstrings-python (API Documentation)
- Auto-generates API reference from Python docstrings
- Supports Google, NumPy, and Sphinx docstring styles
- Automatic cross-references between API objects
- Type annotation display
- Inheritance diagrams
- **Perfect for your component registry system, CLI, core modules**

### 3. mike (Version Management)
- Multiple documentation versions (v0.1.0-alpha.1, latest, stable)
- Each version preserved independently via git branches
- Version selector dropdown in UI
- Essential for tracking changes across releases

## Key Features for Brainsmith

### Diagrams & Visualizations
- **Native Mermaid.js integration** for:
  - Architecture diagrams (DSE tree, segment execution)
  - Flowcharts (compilation pipeline)
  - Sequence diagrams (component registry flow)
  - State diagrams (build stages)
- No external tools needed - diagrams in markdown code blocks

### Code Highlighting
- Multi-language syntax highlighting:
  - Python (core codebase)
  - SystemVerilog/Verilog (RTL kernels)
  - YAML (blueprints)
  - Bash (setup scripts)
  - JSON (configs)

### Advanced Features
- **Content tabs** - Show Docker vs Poetry setup side-by-side
- **Admonitions** - Notes, warnings, tips for complex concepts
- **Code annotations** - Inline explanations in code examples
- **Social cards** - Auto-generated preview images for social sharing
- **Navigation sections** - Collapsible sidebar for large doc sets
- **Search highlighting** - Find content instantly

## Why This Beats Alternatives

### vs Sphinx:
- ✅ Markdown instead of reStructuredText (easier to write/maintain)
- ✅ Instant live reload (no `make html` rebuilds)
- ✅ Modern, beautiful UI out-of-box (Sphinx requires heavy theming)
- ✅ Faster setup and simpler configuration
- ⚠️ mkdocstrings bridges the API doc gap that Sphinx traditionally had

### vs Docusaurus:
- ✅ Python-native tooling (fits your stack)
- ✅ Simpler, no React/Node.js needed
- ✅ Better Python API documentation integration

### vs Read the Docs:
- ✅ GitHub Pages hosting (free, integrated with repo)
- ✅ More modern design aesthetics
- ✅ Better control over customization

## Recommended Plugin Stack

```yaml
# mkdocs.yml
plugins:
  - search              # Built-in search
  - mkdocstrings:       # API docs from docstrings
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
  - mike:               # Version management
      version_selector: true
  - git-revision-date-localized  # Last updated dates
  - minify              # Optimize HTML/CSS/JS

markdown_extensions:
  - pymdownx.superfences:    # Code blocks + Mermaid
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:         # Content tabs
      alternate_style: true
  - pymdownx.highlight       # Code highlighting
  - pymdownx.inlinehilite    # Inline code
  - admonition              # Callout boxes
  - pymdownx.details        # Collapsible admonitions
  - attr_list               # Button styling
  - md_in_html              # Markdown in HTML
  - toc:                    # Table of contents
      permalink: true
```

## Suggested Documentation Structure

```
docs/
├── index.md                      # Landing page
├── getting-started/
│   ├── installation.md          # Setup with Poetry
│   ├── quickstart.md            # BERT quicktest walkthrough
│   └── configuration.md         # Config system guide
├── user-guide/
│   ├── blueprints.md            # Blueprint system
│   ├── design-space.md          # DSE concepts
│   ├── cli-reference.md         # smith/brainsmith CLIs
│   └── examples.md              # Common workflows
├── architecture/
│   ├── overview.md              # High-level architecture
│   ├── component-registry.md    # Registry & decorators
│   ├── segment-execution.md     # DSE tree mechanics
│   └── dataflow-pipeline.md     # ONNX→RTL flow
├── kernel-development/
│   ├── kernel-integrator.md     # RTL kernel creation
│   ├── pragma-reference.md      # SystemVerilog pragmas
│   └── backend-development.md   # Adding backends
├── api-reference/
│   ├── core.md                  # ::: brainsmith.core
│   ├── registry.md              # ::: brainsmith.registry
│   ├── kernels.md               # ::: brainsmith.kernels
│   └── transforms.md            # ::: brainsmith.primitives.transforms
├── tutorials/
│   ├── first-accelerator.md     # End-to-end example
│   ├── custom-kernel.md         # Adding new kernel
│   └── blueprint-authoring.md   # Creating blueprints
└── contributing/
    ├── development-setup.md
    ├── testing.md
    └── code-style.md
```

## Implementation Estimate

### Initial Setup: ~2-4 hours
- Install deps, configure mkdocs.yml
- Set up GitHub Actions for auto-deploy
- Create basic structure and landing page

### Content Migration: ~8-16 hours
- Move existing docs/ content to new structure
- Convert READMEs to proper pages
- Add architecture diagrams with Mermaid

### API Documentation: ~4-8 hours
- Set up mkdocstrings for all modules
- Write module-level docstrings
- Create API reference pages

### Polish & Examples: ~8-16 hours
- Add tutorials with code examples
- Create Mermaid diagrams for key concepts
- Write quickstart guides

**Total: 22-44 hours for comprehensive documentation**

## Resources & References

- [Material for MkDocs Official](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)
- [mike Version Management](https://github.com/jimporter/mike)
- [Real Python MkDocs Tutorial](https://realpython.com/python-project-documentation-with-mkdocs/)
- [MkDocs Catalog of Projects](https://github.com/mkdocs/catalog)

## Example Projects Using This Stack

- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- Typer: https://typer.tiangolo.com/
- Textual: https://textual.textualize.io/
- Prefect: https://docs.prefect.io/
