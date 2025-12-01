#!/usr/bin/env python
"""
Run the ToPy web application.

Usage:
    python -m topy.webapp.run_server [--host HOST] [--port PORT] [--debug]

Example:
    python -m topy.webapp.run_server --port 5000 --debug
"""
import argparse
import os

from topy.webapp import create_app


def main():
    """Run the ToPy web server."""
    parser = argparse.ArgumentParser(description='Run ToPy web application')
    parser.add_argument('--host', default='0.0.0.0', 
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    app = create_app()
    
    print(f"\n🚀 ToPy Web Application")
    print(f"   Running on http://{args.host}:{args.port}")
    print(f"   Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"\n   Press Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
