"""
Utility functions for automation framework
"""

import uuid
from typing import Dict, List, Any, Optional
from .models import AutomationContext, WorkflowConfiguration

def create_automation_context(job: Any) -> AutomationContext:
    """Create automation context from job."""
    from .models import WorkflowStep
    return AutomationContext(
        job=job,
        current_step=WorkflowStep.INITIALIZATION
    )

def validate_workflow_config(config: WorkflowConfiguration) -> bool:
    """Validate workflow configuration."""
    return config.validate()

def generate_workflow_id() -> str:
    """Generate unique workflow ID."""
    return f"workflow_{uuid.uuid4().hex[:12]}"