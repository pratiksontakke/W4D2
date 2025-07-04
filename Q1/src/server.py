"""
Smart Meeting Assistant MCP Server

This is the main MCP server that exposes meeting management tools to AI clients.
It supports both stdio (local) and HTTP+SSE (remote) transports.

The server provides these core tools:
1. create_meeting - Schedule new meetings with conflict detection
2. find_optimal_slots - AI-powered time slot recommendations
3. detect_scheduling_conflicts - Identify scheduling conflicts

Usage:
- Local (stdio): python server.py
- HTTP server: python server.py --http --port 8080
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
import structlog
from dotenv import load_dotenv

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Import our tools and models
try:
    from tools import mcp
    from models import db_manager
    from llm_client import llm_client
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)


async def initialize_server():
    """
    Initialize the MCP server and its dependencies.
    
    This function:
    1. Creates database tables if they don't exist
    2. Checks database connectivity
    3. Validates LLM client configuration
    4. Logs server configuration
    """
    logger.info("Initializing Smart Meeting Assistant MCP Server...")
    
    # Initialize database
    try:
        db_manager.create_tables()
        if db_manager.health_check():
            logger.info("Database initialized successfully")
        else:
            logger.warning("Database health check failed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Check LLM client status
    if llm_client.client:
        logger.info(f"LLM client initialized with provider: {llm_client.provider.value}")
    else:
        logger.warning("LLM client not available - AI features will be limited")
    
    # Log environment configuration
    logger.info("Server configuration", 
                database_url=db_manager.database_url,
                llm_provider=os.getenv("LLM_PROVIDER", "openai"),
                has_openai_key=bool(os.getenv("OPENAI_API_KEY")),
                has_anthropic_key=bool(os.getenv("ANTHROPIC_API_KEY")),
                has_google_key=bool(os.getenv("GOOGLE_API_KEY")))


async def run_stdio_server():
    """
    Run the MCP server in stdio mode for local development.
    This is the default mode used by Cursor and other local AI clients.
    """
    logger.info("Starting MCP server in stdio mode...")
    
    try:
        await initialize_server()
        
        # Run the server
        await mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


async def run_http_server(host: str = "localhost", port: int = 8080):
    """
    Run the MCP server in HTTP+SSE mode for remote access.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    logger.info(f"Starting MCP server in HTTP mode on {host}:{port}...")
    
    try:
        await initialize_server()
        
        # Run the HTTP server
        await mcp.run_http(host=host, port=port)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


def create_sample_env_file():
    """
    Create a sample .env file with all required environment variables.
    """
    env_content = """# Smart Meeting Assistant MCP Server Configuration

# Database Configuration
DATABASE_URL=sqlite:///./data/meetings.db

# LLM Provider Configuration (choose one: openai, anthropic, gemini)
LLM_PROVIDER=openai

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration (if using Anthropic)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini Configuration (if using Google)
GOOGLE_API_KEY=your_google_api_key_here

# Logging Configuration
LOG_LEVEL=INFO

# Server Configuration
MCP_SERVER_NAME=Smart Meeting Assistant
MCP_SERVER_VERSION=1.0.0
"""
    
    env_file = Path(".env.example")
    if not env_file.exists():
        env_file.write_text(env_content)
        logger.info(f"Created sample environment file: {env_file}")
        print(f"\n📄 Created sample environment file: {env_file}")
        print("Please copy this to .env and configure your API keys.")


def main():
    """
    Main entry point for the MCP server.
    Handles command line arguments and starts the appropriate server mode.
    """
    parser = argparse.ArgumentParser(
        description="Smart Meeting Assistant MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                    # Run in stdio mode (default)
  python server.py --http             # Run HTTP server on localhost:8080
  python server.py --http --port 3000 # Run HTTP server on port 3000
  python server.py --create-env        # Create sample .env file
        """
    )
    
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run in HTTP+SSE mode instead of stdio mode"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to in HTTP mode (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on in HTTP mode (default: 8080)"
    )
    
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create a sample .env file and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Smart Meeting Assistant MCP Server 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_env:
        create_sample_env_file()
        return
    
    # Print startup banner
    print("🤖 Smart Meeting Assistant MCP Server")
    print("=====================================")
    print("AI-powered meeting scheduling with conflict detection")
    print()
    
    try:
        if args.http:
            # Run HTTP server
            print(f"🌐 Starting HTTP server on {args.host}:{args.port}")
            print(f"📡 MCP endpoint: http://{args.host}:{args.port}/mcp")
            print("🔗 Connect your AI client to this endpoint")
            print()
            asyncio.run(run_http_server(args.host, args.port))
        else:
            # Run stdio server
            print("📡 Starting stdio server (local mode)")
            print("🔗 This server will communicate via stdin/stdout")
            print("💡 Cursor and other local AI clients will auto-detect this server")
            print()
            asyncio.run(run_stdio_server())
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
