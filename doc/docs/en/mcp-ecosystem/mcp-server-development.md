# MCP Server Development Guide

This guide walks you through building your own MCP server with Python and the FastMCP framework, then connecting it to the Nexent platform.

## 🌐 Language Support

The MCP protocol provides SDKs for multiple programming languages:

- **Python** ⭐ (recommended)
- **TypeScript**
- **Java**
- **Go**
- **Rust**
- Any other language that implements the MCP protocol

### Why Do We Recommend Python?

We use **Python** for the examples in this guide because it offers:

- ✅ **Beginner-friendly syntax**: concise code that is easy to read
- ✅ **Rich ecosystem**: frameworks like FastMCP remove most boilerplate
- ✅ **Rapid prototyping**: you can stand up a working server in minutes
- ✅ **Mature libraries**: thousands of third-party packages are available

If you are already comfortable in another language, feel free to use the corresponding MCP SDK. For a first MCP server, however, Python gives you the smoothest experience.

## 📋 Prerequisites

Install FastMCP before you start coding:

```bash
pip install fastmcp
```

## 🚀 Quick Start

### Minimal Example

The snippet below creates a simple string utility server with FastMCP:

```python
from fastmcp import FastMCP

# Create an MCP server instance
mcp = FastMCP(name="String MCP Server")

@mcp.tool(
    name="calculate_string_length",
    description="Calculate the length of a string"
)
def calculate_string_length(text: str) -> int:
    return len(text)

@mcp.tool(
    name="to_uppercase",
    description="Convert text to uppercase"
)
def to_uppercase(text: str) -> str:
    return text.upper()

@mcp.tool(
    name="to_lowercase",
    description="Convert text to lowercase"
)
def to_lowercase(text: str) -> str:
    return text.lower()

if __name__ == "__main__":
    # Start with SSE transport
    mcp.run(transport="sse", port=8000)
```

### Run the Server

Save the code as `mcp_server.py` and execute:

```bash
python mcp_server.py
```

You should see the server start successfully with the endpoint `http://127.0.0.1:8000/sse`.

## 🔌 Integrate MCP Services with Nexent

Once your MCP server is up, connect it to Nexent:

### Step 1: Start the MCP Server

Keep the server process running and note the public endpoint (for example `http://127.0.0.1:8000/sse`).

### Step 2: Register the MCP Service in Nexent

1. Open the **[Agent Development](../user-guide/agent-development.md)** page.
2. On the “Select Agent Tools” tab, click **MCP Configuration** on the right.
3. Enter the server name and server URL.
   - ⚠️ **Important**:
     1. The server name must contain only letters and digits—no spaces or other symbols.
     2. When Nexent runs inside Docker and the MCP server runs on the host, replace `127.0.0.1` with `host.docker.internal`, for example `http://host.docker.internal:8000`.
4. Click **Add** to finish the registration.

### Step 3: Use the MCP Tool

During agent creation or editing, the newly registered MCP tool appears in the tool list and can be attached to any agent.

## 🔧 Wrap Existing Workloads

To expose existing business logic as MCP tools, call your internal APIs or libraries inside the tool functions.

### Example: Wrap a REST API

```python
from fastmcp import FastMCP
import requests

# Create an MCP server instance
mcp = FastMCP("Course Statistics Server")

@mcp.tool(
    name="get_course_statistics",
    description="Get course statistics such as average, max, min, and total students"
)
def get_course_statistics(course_id: str) -> str:
    api_url = "https://your-school-api.com/api/courses/statistics"
    response = requests.get(api_url, params={"course_id": course_id})

    if response.status_code == 200:
        data = response.json()
        stats = data.get("statistics", {})
        return (
            f"Course {course_id} statistics:\n"
            f"Average: {stats.get('average', 'N/A')}\n"
            f"Max: {stats.get('max', 'N/A')}\n"
            f"Min: {stats.get('min', 'N/A')}\n"
            f"Total Students: {stats.get('total_students', 'N/A')}"
        )
    return f"API request failed: {response.status_code}"

if __name__ == "__main__":
    # Start with SSE transport
    mcp.run(transport="sse", port=8000)
```

### Example: Wrap an Internal Module

```python
from fastmcp import FastMCP
from your_school_module import query_course_statistics

# Create an MCP server instance
mcp = FastMCP("Course Statistics Server")

@mcp.tool(
    name="get_course_statistics",
    description="Get course statistics such as average, max, min, and total students"
)
def get_course_statistics(course_id: str) -> str:
    try:
        stats = query_course_statistics(course_id)
        return (
            f"Course {course_id} statistics:\n"
            f"Average: {stats.get('average', 'N/A')}\n"
            f"Max: {stats.get('max', 'N/A')}\n"
            f"Min: {stats.get('min', 'N/A')}\n"
            f"Total Students: {stats.get('total_students', 'N/A')}"
        )
    except Exception as exc:
        return f"Failed to query statistics: {exc}"

if __name__ == "__main__":
    # Start with SSE transport
    mcp.run(transport="sse", port=8000)
```

## 📚 Additional Resources

### Python

- [FastMCP Documentation](https://github.com/modelcontextprotocol/python-sdk) (used throughout this guide)

### Other Languages

- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Java SDK](https://github.com/modelcontextprotocol/java-sdk)
- [MCP Go SDK](https://github.com/modelcontextprotocol/go-sdk)
- [MCP Rust SDK](https://github.com/modelcontextprotocol/rust-sdk)

### General References

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Nexent Agent Development Guide](../user-guide/agent-development.md)
- [MCP Tool Ecosystem Overview](./overview.md)

## 🆘 Need Help?

If you run into issues while developing MCP servers:

1. Check the **[FAQ](../getting-started/faq.md)**
2. Ask questions in [GitHub Discussions](https://github.com/ModelEngine-Group/nexent/discussions)
3. Review sample servers on the [ModelScope MCP Marketplace](https://www.modelscope.cn/mcp)

