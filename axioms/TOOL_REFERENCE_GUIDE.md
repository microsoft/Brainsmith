# Tool Reference Guide

This document provides a comprehensive overview of all available tools and their syntax for working with the Roo development environment.

## File Operations

### read_file
**Purpose**: Read contents of one or more files (up to 15 files simultaneously)
**Returns**: Line-numbered content for easy reference
**Supports**: Text extraction from PDF and DOCX files

```xml
<read_file>
<args>
  <file>
    <path>src/main.py</path>
  </file>
  <file>
    <path>config/settings.json</path>
  </file>
</args>
</read_file>
```

### write_to_file
**Purpose**: Create new files or completely rewrite existing files
**Note**: Overwrites existing files, creates directories as needed

```xml
<write_to_file>
<path>src/new_file.py</path>
<content>
def hello_world():
    print("Hello, World!")
### insert_content
**Purpose**: Add new lines of content to a file without modifying existing content
**Use Cases**: Adding imports, functions, configuration blocks
**Parameters**:
- `path`: File path relative to workspace
- `line`: Line number to insert before (0 = append to end)
- `content`: Content to insert

```xml
<insert_content>
<path>example/file.py</path>
<line>1</line>
<content>
import logging
from typing import Dict, List
Returns**: Context-rich results with surrounding lines

```xml
<search_files>
<path>src/</path>
<regex>function\s+\w+</regex>
<file_pattern>*.py</file_pattern>
</search_files>
```

### list_code_definition_names
**Purpose**: List definition names (classes, functions, methods) from source code
**Scope**: Single file or all files in a directory

```xml
<list_code_definition_names>
<path>src/main.py</path>
</list_code_definition_names>
```

## System Operations

### execute_command
**Purpose**: Execute CLI commands on the system
**Note**: Each command runs in a new terminal instance
**Parameters**:
- `command`: The CLI command to execute
- `cwd`: Optional working directory

```xml
<execute_command>
<command>npm install</command>
<cwd>/path/to/project</cwd>
</execute_command>
```

## MCP (Model Context Protocol) Tools

### use_mcp_tool
**Purpose**: Use tools provided by connected MCP servers
**Requirements**: Server must be connected and tool must be available

```xml
<use_mcp_tool>
<server_name>weather-server</server_name>
<tool_name>get_forecast</tool_name>
<arguments>
{
  "city": "San Francisco",
  "days": 5
}
</arguments>
</use_mcp_tool>
```

### access_mcp_resource
**Purpose**: Access resources from connected MCP servers
**Use Cases**: Files, API responses, system information

```xml
<access_mcp_resource>
<server_name>file-server</server_name>
<uri>file://path/to/resource</uri>
</access_mcp_resource>
```

## Interaction Tools

### ask_followup_question
**Purpose**: Ask the user for additional information needed to complete tasks
**Best Practice**: Provide 2-4 suggested answers

```xml
<ask_followup_question>
<question>Which framework would you like to use?</question>
<follow_up>
<suggest>React with TypeScript</suggest>
<suggest>Vue.js with JavaScript</suggest>
<suggest>Angular with TypeScript</suggest>
</follow_up>
</ask_followup_question>
```

## Task Management Tools

### fetch_instructions
**Purpose**: Get detailed instructions for specific tasks
**Available Tasks**: create_mcp_server, create_mode

```xml
<fetch_instructions>
<task>create_mcp_server</task>
</fetch_instructions>
```

### new_task
**Purpose**: Create a new task instance in a specified mode

```xml
<new_task>
<mode>code</mode>
<message>Implement authentication system</message>
</new_task>
```

### switch_mode
**Purpose**: Switch to a different operational mode
**Available Modes**: code, debug, architect, ask, orchestrator

```xml
<switch_mode>
<mode_slug>debug</mode_slug>
<reason>Need to troubleshoot application errors</reason>
</switch_mode>
```

### attempt_completion
**Purpose**: Present the final result of completed work
**Note**: Can only be used after confirming previous tool uses were successful

```xml
<attempt_completion>
<result>
Successfully implemented the user authentication system with login, logout, and session management.
</result>
<command>python app.py</command>
</attempt_completion>
```

## Important Usage Guidelines

1. **One Tool Per Message**: Only one tool can be used per message
2. **Wait for Confirmation**: Always wait for user confirmation after each tool use
3. **File Path Requirements**: All file paths must be relative to workspace directory
4. **Error Handling**: Check tool results before proceeding to next steps
5. **Efficiency**: Read multiple related files simultaneously when possible (up to 15 files)
6. **Best Practices**: Use appropriate tools for specific tasks (e.g., list_files instead of ls command)

## Tool Selection Strategy

- **File Reading**: Use `read_file` for examining existing code
- **File Creation**: Use `write_to_file` for new files
- **File Modification**: Use `apply_diff` for targeted changes, `search_and_replace` for pattern replacements
- **Code Analysis**: Use `search_files` for finding patterns, `list_code_definition_names` for structure overview
- **System Tasks**: Use `execute_command` for CLI operations
- **User Interaction**: Use `ask_followup_question` when clarification is needed