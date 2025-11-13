## Description

<!-- Provide a clear and concise description of your changes -->

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] CI/CD or infrastructure change
- [ ] Refactoring (no functional changes)

## Related Issues

<!-- Link to related issues using #issue_number -->

Fixes #
Related to #

## Changes Made

<!-- Provide a bulleted list of the changes made in this PR -->

-
-
-

## Testing

<!-- Describe the tests you ran and how to reproduce -->

### Test Environment
- Python version:
- Poetry version:
- Operating System:

### Tests Performed
- [ ] I have tested this locally
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have run `poetry run pytest -m 'unit or integration'` successfully

### Test Coverage
<!-- Describe which parts of the code are covered by tests -->

## Documentation

- [ ] I have updated the documentation to reflect my changes
- [ ] I have updated docstrings for modified functions/classes
- [ ] I have added/updated examples if applicable
- [ ] Documentation builds successfully (`mkdocs build`)

## Code Quality

- [ ] My code follows the style guidelines of this project (`ruff check .` passes)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have removed any debugging code (print statements, breakpoints, etc.)

## Breaking Changes

<!-- If this PR introduces breaking changes, describe them here and provide migration instructions -->

**Does this PR introduce breaking changes?**
- [ ] Yes
- [ ] No

<!-- If yes, describe the breaking changes and migration path: -->

## Security Considerations

<!-- If this PR modifies security-sensitive files, explain why the changes are safe -->

- [ ] This PR does **not** modify `.github/workflows/`, `.github/actions/`, Docker configs, or shell scripts
- [ ] OR: I have explained why these changes are necessary and safe (see below)

<!-- If you checked the second option, explain here: -->

---

## For Maintainers

**Before adding `safe-to-test` label:**

### Security Review
- [ ] Code review completed (especially security-sensitive files)
- [ ] No suspicious network calls or data exfiltration attempts
- [ ] No attempts to access secrets or credentials
- [ ] No hardcoded sensitive information (API keys, passwords, IPs)
- [ ] Docker/workflow changes reviewed by @microsoft/brainsmith team (if applicable)

### Code Review
- [ ] Code follows project conventions and style
- [ ] Tests are adequate and meaningful
- [ ] Documentation is clear and accurate
- [ ] No unnecessary dependencies added
- [ ] Error handling is appropriate

### CI/CD
- [ ] Ready to trigger CI on self-hosted runners
- [ ] Expected test duration is reasonable
- [ ] Artifacts and cleanup are properly configured

**Notes for reviewers:**
<!-- Add any additional notes for reviewers -->
