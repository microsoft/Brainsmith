# Contributing to Brainsmith

## Quick Start

**Prerequisites & Installation**: See [README.md](README.md#prerequisites)

**For Contributors**:

```bash
# After installation, verify development setup
pytest -m 'unit or integration' -v
ruff check .

# Install documentation dependencies
poetry install --with docs
```

## CI Approval Process

**We use self-hosted runners** with expensive Xilinx licenses. External PRs require manual approval:

1. Submit PR → CI shows "skipped"
2. Maintainer reviews → adds `safe-to-test` label
3. CI runs (pytest + hardware tests)
4. Label auto-removes
5. New commits? Need new label

## Files That Need Extra Review

Changes to these trigger mandatory @microsoft/brainsmith review (CODEOWNERS enforced) for security:

- `.github/workflows/`, `.github/actions/` - CI/CD
- `docker/`, `Dockerfile`, `*.sh` - Container/scripts
- `brainsmith/_internal/io/dependencies.py` - Dependency definitions

## What We Look For

**Required**:
- Tests for new features/fixes
- Documentation for user-facing changes
- Code style matches existing patterns (`ruff check .`)

**Never**:
- Hardcoded secrets or credentials
- Network requests in tests (unless mocked)
- Obfuscated code (base64, eval, exec)

## Contributing Kernels

We actively encourage kernel contributions—each new kernel exponentially increases Brainsmith's value by expanding the range of models it can compile. An automated kernel validation and feature tagging system is currently work-in-progress and will significantly simplify kernel contribution in a future release.

For now, kernel PRs require manual schema validation and feature compatibility review.

## Development Commands

```bash
# Tests
pytest -m 'unit or integration' -v  # Fast
pytest tests/ -v                     # Full suite
pytest --cov=brainsmith --cov-report=html

# Linting
ruff check .

# Docs
poetry install --with docs
mkdocs serve
```

## Security

Found a vulnerability? Report through [Microsoft Security Response Center](https://msrc.microsoft.com/create-report), not public issues.

## Help

- Questions: [GitHub Discussions](https://github.com/microsoft/brainsmith/discussions)
<<<<<<< HEAD
- Bugs: [GitHub Issues](https://github.com/microsoft/Brainsmith/issues)
- Docs: [microsoft.github.io/brainsmith](https://microsoft.github.io/brainsmith/)
||||||| 655cd8e0
- Bugs: [GitHub Issues](https://github.com/microsoft/brainsmith/issues)
- Docs: [microsoft.github.io/brainsmith](https://microsoft.github.io/brainsmith/)
=======
- Bugs: [GitHub Issues](https://github.com/microsoft/Brainsmith/issues)
- Docs: [microsoft.github.io/brainsmith](https://microsoft.github.io/Brainsmith/)
>>>>>>> develop

## Code of Conduct

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)

---

Thank you for contributing!
