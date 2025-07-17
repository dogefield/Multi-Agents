# Claude Setup Guide

## Current Status
✅ Claude API client installed  
✅ `claude` command created  
✅ Added to PATH  
❌ API key not set (required)  
❌ WSL features need restart (optional)  

## Quick Start

### 1. Get Your API Key
- Go to: https://console.anthropic.com/
- Sign in with your Claude subscription
- Get your API key from the API Keys section

### 2. Set Your API Key (choose one method)

**Temporary (current session only):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-YOUR-KEY-HERE"
```

**Permanent (recommended):**
```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-api03-YOUR-KEY-HERE", [EnvironmentVariableTarget]::User)
```

### 3. Use Claude

**Quick message:**
```powershell
claude "What is the capital of France?"
```

**Multi-line message (interactive mode):**
```powershell
claude
# Type your message, press Enter twice to send
```

**From any directory:**
```powershell
cd C:\anywhere
claude "Hello Claude!"
```

## Alternative Ways to Use Claude

### 1. Web Interface (Easiest)
- https://claude.ai - Chat with Claude in your browser

### 2. Python Script
```python
from anthropic import Anthropic

client = Anthropic(api_key="your-key-here")
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### 3. WSL/Linux (After Restart)
Once you restart your computer:
```bash
wsl -d Ubuntu
# Install in WSL:
pip3 install anthropic
```

## Troubleshooting

**"claude is not recognized"**
- Close and reopen PowerShell/Terminal
- Or use: `.\claude.bat` from this directory

**"API key not set"**
- Make sure you set the ANTHROPIC_API_KEY environment variable

**"WSL not working"**
- Restart your computer to complete WSL installation

## API Pricing
Check current pricing at: https://www.anthropic.com/pricing 