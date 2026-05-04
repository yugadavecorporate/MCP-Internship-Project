import os
import sys
import sqlite3
import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP server
mcp = FastMCP("Groq_Chatbot_MCP_Server")

DB_PATH = "local_data.db"
SANDBOX_DIR = Path.cwd().resolve()

def init_db():
    """Initialize the SQLite database with some dummy data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create a dummy users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    
    # Check if data already exists to avoid duplicate inserts on restarts
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        sample_users = [
            ("alice_admin", "alice@example.com", "admin"),
            ("bob_user", "bob@example.com", "user"),
            ("charlie_dev", "charlie@example.com", "developer")
        ]
        cursor.executemany("INSERT INTO users (username, email, role) VALUES (?, ?, ?)", sample_users)
        conn.commit()
    
    conn.close()

# Run DB initialization on module load
init_db()

@mcp.tool()
def read_local_file(filepath: str) -> str:
    """
    Read the contents of a local file.
    Only allows reading files within the current working directory.
    
    Args:
        filepath: The path to the file to be read.
    """
    try:
        # Resolve the requested path to an absolute path
        requested_path = Path(filepath).resolve()
        
        # Check if the requested path is within the sandbox directory (path traversal protection)
        if not str(requested_path).startswith(str(SANDBOX_DIR)):
            return "Error: Access denied. File is outside the allowed sandbox directory."
            
        if not requested_path.is_file():
            return f"Error: File not found at {requested_path}"
            
        with open(requested_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def query_local_database(sql_query: str) -> str:
    """
    Execute a SQL query against the local SQLite database.
    Enforces a read-only connection to prevent modifications (DROP, DELETE, UPDATE, etc.).
    
    Args:
        sql_query: The SQL query string to execute.
    """
    try:
        # Prevent obvious write operations through basic string checking just in case
        lower_query = sql_query.lower()
        forbidden_keywords = ['insert ', 'update ', 'delete ', 'drop ', 'create ', 'alter ', 'replace ']
        if any(keyword in lower_query for keyword in forbidden_keywords):
            return "Error: Only SELECT queries are allowed."
            
        # Connect to SQLite in read-only mode using a file URI string
        db_uri = f"file:{Path(DB_PATH).resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        if not rows:
            return "Query executed successfully. No results found."
            
        # Format results as a list of dictionaries
        results = [dict(row) for row in rows]
        conn.close()
        return str(results)
        
    except sqlite3.OperationalError as e:
        return f"Database Error: {str(e)}"
    except Exception as e:
        return f"Error executing query: {str(e)}"

@mcp.tool()
def fetch_web_api(url: str, method: str = "GET") -> str:
    """
    Make an HTTP request to the provided URL and return the response.
    
    Args:
        url: The web URL to fetch data from.
        method: The HTTP method to use (defaults to GET).
    """
    try:
        method = method.upper()
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
            return f"Error: Unsupported HTTP method '{method}'"
            
        # Using httpx for making synchronous web requests
        with httpx.Client(timeout=10.0) as client:
            response = client.request(method, url)
            response.raise_for_status()  # Raise exception for 4xx and 5xx status codes
            
            # Try to return JSON if possible, otherwise plain text
            try:
                return str(response.json())
            except ValueError:
                return response.text
                
    except httpx.TimeoutException:
        raise ValueError("Error: Request timed out.")
    except httpx.HTTPStatusError as e:
        return f"HTTP Error: Received status code {e.response.status_code}."
    except Exception as e:
        return f"Error making web request: {str(e)}"

if __name__ == "__main__":
    # Start the FastMCP server using standard I/O for communication with the host client
    mcp.run()
