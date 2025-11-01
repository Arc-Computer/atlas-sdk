import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

server = Server("file-operations")

WORKSPACE_DIR = (Path(__file__).parent / "sample_workspace").resolve()


def validate_path(path: str, workspace: Path = WORKSPACE_DIR) -> Path | None:
    try:
        resolved = (workspace / path).resolve()
        if resolved.is_relative_to(workspace):
            return resolved
    except (ValueError, OSError):
        pass
    return None


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="read_file",
            description="Read the contents of a file. Returns the file content as a string.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read (relative or absolute)",
                    }
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write (relative or absolute)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        types.Tool(
            name="list_files",
            description="List all files and directories in a given directory. Returns a list of file names.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The directory path to list files from",
                    }
                },
                "required": ["directory"],
            },
        ),
        types.Tool(
            name="search_content",
            description="Search for a pattern in all files within a directory. Returns matches with file paths and line numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for",
                    },
                    "directory": {
                        "type": "string",
                        "description": "The directory to search in",
                    },
                },
                "required": ["pattern", "directory"],
            },
        ),
        types.Tool(
            name="run_command",
            description="Execute a shell command in a safe, sandboxed manner. Returns command output.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (restricted to safe operations)",
                    }
                },
                "required": ["command"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if arguments is None:
        arguments = {}

    try:
        if name == "read_file":
            path = arguments.get("path", "")
            file_path = validate_path(path)
            if file_path is None:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: Path outside workspace sandbox: {path}",
                    )
                ]
            if not file_path.exists():
                return [
                    types.TextContent(
                        type="text", text=f"Error: File not found: {path}"
                    )
                ]
            content = file_path.read_text(encoding="utf-8")
            return [
                types.TextContent(
                    type="text", text=f"File content of {path}:\n\n{content}"
                )
            ]

        elif name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            file_path = validate_path(path)
            if file_path is None:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: Path outside workspace sandbox: {path}",
                    )
                ]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully wrote {len(content)} characters to {path}",
                )
            ]

        elif name == "list_files":
            directory = arguments.get("directory", ".")
            dir_path = validate_path(directory)
            if dir_path is None:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: Path outside workspace sandbox: {directory}",
                    )
                ]
            if not dir_path.exists():
                return [
                    types.TextContent(
                        type="text", text=f"Error: Directory not found: {directory}"
                    )
                ]
            if not dir_path.is_dir():
                return [
                    types.TextContent(
                        type="text", text=f"Error: Not a directory: {directory}"
                    )
                ]

            files = []
            for item in dir_path.iterdir():
                prefix = "[DIR]" if item.is_dir() else "[FILE]"
                files.append(f"{prefix} {item.name}")

            files_list = "\n".join(sorted(files))
            return [
                types.TextContent(
                    type="text", text=f"Contents of {directory}:\n\n{files_list}"
                )
            ]

        elif name == "search_content":
            pattern = arguments.get("pattern", "")
            directory = arguments.get("directory", ".")
            dir_path = validate_path(directory)
            if dir_path is None:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: Path outside workspace sandbox: {directory}",
                    )
                ]
            if not dir_path.exists():
                return [
                    types.TextContent(
                        type="text", text=f"Error: Directory not found: {directory}"
                    )
                ]

            try:
                regex = re.compile(pattern)
            except re.error as e:
                return [
                    types.TextContent(
                        type="text", text=f"Error: Invalid regex pattern: {e}"
                    )
                ]

            matches = []
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        for line_num, line in enumerate(content.splitlines(), 1):
                            if regex.search(line):
                                matches.append(
                                    {
                                        "file": str(file_path.relative_to(dir_path)),
                                        "line": line_num,
                                        "content": line.strip(),
                                    }
                                )
                    except (UnicodeDecodeError, PermissionError):
                        continue

            if not matches:
                return [
                    types.TextContent(
                        type="text", text=f"No matches found for pattern: {pattern}"
                    )
                ]

            result_text = f"Found {len(matches)} matches for '{pattern}':\n\n"
            for match in matches[:50]:
                result_text += f"{match['file']}:{match['line']}: {match['content']}\n"

            if len(matches) > 50:
                result_text += f"\n... and {len(matches) - 50} more matches"

            return [types.TextContent(type="text", text=result_text)]

        elif name == "run_command":
            command = arguments.get("command", "")

            safe_commands = {"ls", "pwd", "echo", "cat", "grep", "wc", "head", "tail"}

            try:
                tokens = shlex.split(command)
            except ValueError as e:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: Invalid command syntax: {e}",
                    )
                ]

            if not tokens or tokens[0] not in safe_commands:
                cmd_name = tokens[0] if tokens else ""
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: Command '{cmd_name}' is not allowed. Safe commands: {', '.join(sorted(safe_commands))}",
                    )
                ]

            cmd_name = tokens[0]
            sanitized_args: list[str] = []
            pattern_consumed = False

            for arg in tokens[1:]:
                if arg.startswith("-"):
                    sanitized_args.append(arg)
                    continue
                if cmd_name in {"echo", "pwd"}:
                    sanitized_args.append(arg)
                    continue
                if cmd_name == "grep" and not pattern_consumed:
                    sanitized_args.append(arg)
                    pattern_consumed = True
                    continue

                resolved = validate_path(arg)
                if resolved is None:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Error: Path outside workspace sandbox: {arg}",
                        )
                    ]
                try:
                    relative = resolved.relative_to(WORKSPACE_DIR)
                except ValueError:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Error: Path outside workspace sandbox: {arg}",
                        )
                    ]
                sanitized_args.append(str(relative))

            try:
                result = subprocess.run(
                    [cmd_name, *sanitized_args],
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=WORKSPACE_DIR,
                )
                output = result.stdout if result.returncode == 0 else result.stderr
                return [
                    types.TextContent(
                        type="text",
                        text=f"Command output (exit code {result.returncode}):\n\n{output}",
                    )
                ]
            except subprocess.TimeoutExpired:
                return [
                    types.TextContent(
                        type="text", text="Error: Command execution timed out (10s limit)"
                    )
                ]
            except Exception as e:
                return [
                    types.TextContent(type="text", text=f"Error executing command: {e}")
                ]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="file-operations",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import anyio

    anyio.run(main)
