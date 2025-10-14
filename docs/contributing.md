# Contributing to Brainsmith

We welcome contributions! This guide will help you get started.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/brainsmith.git
   cd brainsmith
   ```
3. Run setup:
   ```bash
   ./setup-venv.sh
   source .venv/bin/activate
   ```
4. Configure Vivado paths:
   ```bash
   brainsmith config init
   # Edit ~/.brainsmith/config.yaml
   ```

## Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/integration/test_plugin_system.py

# With coverage
pytest tests/ --cov=brainsmith.core
```

## Code Style

We use minimal linting rules during the alpha phase:

```bash
# Check code
ruff check brainsmith/ tests/

# Auto-fix issues
ruff check --fix brainsmith/ tests/

# Format code
ruff format brainsmith/ tests/
```

### Style Guidelines

- Line length: 100 characters
- Python 3.10+ required
- Docstring style: Google format preferred

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes

3. Add tests for new functionality

4. Run tests and linting:
   ```bash
   pytest tests/
   ruff check brainsmith/ tests/
   ```

5. Commit with descriptive messages:
   ```bash
   git commit -m "Add feature X that does Y"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/my-feature
   ```

7. Open a Pull Request on GitHub

## PR Guidelines

- Describe what your PR does and why
- Reference any related issues
- Ensure all tests pass
- Add documentation for new features
- Keep PRs focused and reasonably sized

## Documentation

When adding features, update documentation:

- Add/update docstrings
- Update relevant markdown docs in `docs/`
- Add examples if applicable

Build docs locally:

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000

## Questions?

- Open an [issue](https://github.com/microsoft/brainsmith/issues)
- Start a [discussion](https://github.com/microsoft/brainsmith/discussions)

## Code of Conduct

Please be respectful and follow our [Code of Conduct](https://github.com/microsoft/brainsmith/blob/main/CODE_OF_CONDUCT.md).
