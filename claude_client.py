#!/usr/bin/env python3
"""
Simple Claude CLI - Talk to Claude from your terminal
"""
import os
import sys
from anthropic import Anthropic

def main():
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set!")
        print("\nTo set it:")
        print("  Windows (PowerShell): $env:ANTHROPIC_API_KEY = 'your-api-key-here'")
        print("  Windows (CMD): set ANTHROPIC_API_KEY=your-api-key-here")
        print("\nGet your API key from: https://console.anthropic.com/")
        sys.exit(1)
    
    # Initialize Claude client
    client = Anthropic(api_key=api_key)
    
    # Get user input
    if len(sys.argv) > 1:
        # Use command line arguments as the message
        user_message = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("Claude CLI - Type your message (press Enter twice to send):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        user_message = "\n".join(lines[:-1])  # Remove last empty line
    
    if not user_message.strip():
        print("No message provided!")
        sys.exit(1)
    
    print("\nClaude is thinking...\n")
    
    try:
        # Send message to Claude
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        # Print Claude's response
        print("Claude:", response.content[0].text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 