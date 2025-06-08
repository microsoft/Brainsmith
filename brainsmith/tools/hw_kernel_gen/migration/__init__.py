"""
Migration utilities for transitioning to enhanced architecture.

This module provides tools and utilities to help migrate existing code,
configurations, and workflows from the legacy architecture to the new
Week 3 enhanced system.
"""

from .migration_utilities import (
    MigrationStatus,
    MigrationReport,
    ConfigurationMigrator,
    WorkflowMigrator,
    DataStructureMigrator,
    migrate_configuration,
    analyze_workflow,
    create_migration_guide
)

__all__ = [
    "MigrationStatus",
    "MigrationReport", 
    "ConfigurationMigrator",
    "WorkflowMigrator",
    "DataStructureMigrator",
    "migrate_configuration",
    "analyze_workflow",
    "create_migration_guide"
]