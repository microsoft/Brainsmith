# Kernel Contributions

This directory is designed for stakeholder-contributed kernel implementations.

## Structure
- **Custom Operations**: Add FINN custom operation implementations
- **Hardware Sources**: Add HLS/RTL kernel sources  
- **Kernel Definitions**: Add new kernel YAML configurations

## Adding New Kernels
1. Create a subdirectory for your kernel family (e.g., `my_company_kernels/`)
2. Include kernel YAML definitions following the existing format
3. Add corresponding custom operation implementations
4. Include hardware sources (HLS/RTL) as needed
5. Update the kernel registry for automatic discovery

## Guidelines
- Follow existing naming conventions
- Include comprehensive documentation
- Ensure compatibility with the FINN compilation flow
- Add appropriate tests for your kernels