#!/bin/bash
# Script to update test imports from rtl_data to new types

# Update rtl_data imports to types.rtl
find /home/tafk/dev/brainsmith-1/tests -name "*.py" -type f -exec sed -i \
  -e 's/from brainsmith\.tools\.kernel_integrator\.rtl_parser\.rtl_data import Parameter/from brainsmith.tools.kernel_integrator.types.rtl import Parameter/g' \
  -e 's/from brainsmith\.tools\.kernel_integrator\.rtl_parser\.rtl_data import Port/from brainsmith.tools.kernel_integrator.types.rtl import Port/g' \
  -e 's/from brainsmith\.tools\.kernel_integrator\.rtl_parser\.rtl_data import PortGroup/from brainsmith.tools.kernel_integrator.types.rtl import PortGroup/g' \
  -e 's/from brainsmith\.tools\.kernel_integrator\.rtl_parser\.rtl_data import PragmaType/from brainsmith.tools.kernel_integrator.types.rtl import PragmaType/g' \
  -e 's/from brainsmith\.tools\.kernel_integrator\.rtl_parser\.rtl_data import ProtocolValidationResult/from brainsmith.tools.kernel_integrator.types.rtl import ProtocolValidationResult/g' \
  {} \;

# Update PortDirection imports to types.core
find /home/tafk/dev/brainsmith-1/tests -name "*.py" -type f -exec sed -i \
  's/from brainsmith\.tools\.kernel_integrator\.rtl_parser\.rtl_data import PortDirection/from brainsmith.tools.kernel_integrator.types.core import PortDirection/g' \
  {} \;

# Update combined imports
find /home/tafk/dev/brainsmith-1/tests -name "*.py" -type f -exec sed -i \
  's/from brainsmith\.tools\.kernel_integrator\.rtl_parser\.rtl_data import (/from brainsmith.tools.kernel_integrator.types.rtl import (/g' \
  {} \;

echo "Test imports updated"