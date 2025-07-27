#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

from brainsmith.core.plugins.registry import list_steps
print("Available steps:", list_steps())