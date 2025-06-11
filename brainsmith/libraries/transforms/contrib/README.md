# Transform Contributions

This directory is designed for stakeholder-contributed transformation implementations.

## Structure
- **Pipeline Steps**: Add new transformation steps for the compilation pipeline
- **Operations**: Add new transformation operations for models
- **Optimization Passes**: Add custom optimization strategies

## Adding New Transforms
1. Create a subdirectory for your transform family (e.g., `my_company_transforms/`)
2. Include transformation step implementations following existing patterns
3. Add transformation operations for model manipulation
4. Include comprehensive documentation and examples
5. Update transformation registries for automatic discovery

## Guidelines
- Follow the existing transformation API patterns
- Ensure compatibility with FINN model formats
- Include appropriate error handling and validation
- Add comprehensive tests for your transformations
- Document any dependencies or requirements