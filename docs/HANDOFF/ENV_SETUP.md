# Environment Variables Setup

Environment variables are used to store sensitive information (like API keys) outside the code.  
This prevents exposing personal credentials in the codebase or Git repository.

---

## Quick setup

```bash
# Set environment variables for current terminal session
# Replace "sk-...your_key..." with your actual API key

# Windows PowerShell
$env:OPENAI_API_KEY="sk-...your_key..."
$env:LLM_BACKEND="openai"
$env:LLM_MODEL="gpt-5-mini"
$env:LLM_TEMPERATURE="0.3"

# Windows CMD
set OPENAI_API_KEY=sk-...your_key...
set LLM_BACKEND=openai
set LLM_MODEL=gpt-5-mini
set LLM_TEMPERATURE=0.3

# macOS / Linux / WSL
export OPENAI_API_KEY="sk-...your_key..."
export LLM_BACKEND="openai"
export LLM_MODEL="gpt-5-mini"
export LLM_TEMPERATURE="0.3"

```

## Tip: 
For permanent settings, add these lines to your shell profile file (e.g., ~/.bashrc, ~/.zshrc, or use setx in Windows).
