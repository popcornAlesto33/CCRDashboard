#!/usr/bin/env python3
"""Migrate Scripts/.env from old LLM_* format to new multi-provider format.

Run once: python3 Scripts/migrate_env.py
Creates Scripts/.env.new — review it, then rename to .env
"""
import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Read current values
current = {}
with open(env_path, "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()

# Build new .env content
new_env = f"""# ================================================================
# Provider: Gemini (default)
# ================================================================
GEMINI_API_KEY={current.get('LLM_API_KEY', 'your-gemini-key-here')}
GEMINI_BASE_URL={current.get('LLM_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/')}

# ================================================================
# Provider: OpenAI
# ================================================================
OPENAI_API_KEY=your-openai-key-here
# OPENAI_BASE_URL=https://api.openai.com/v1  # default, uncomment to override

# ================================================================
# Provider: Anthropic
# ================================================================
ANTHROPIC_API_KEY=your-anthropic-key-here
# ANTHROPIC_BASE_URL=https://api.anthropic.com/v1/  # default, uncomment to override

# ================================================================
# Default Provider & Models
# ================================================================
# Which provider to use when --provider flag is not passed
LLM_PROVIDER=gemini

# Override default models per provider (optional)
# When using --provider, these are ignored in favor of provider defaults
# Gemini defaults:    gemini-2.5-pro (reasoning), gemini-2.5-flash (classification)
# OpenAI defaults:    gpt-5 (reasoning), gpt-4o-mini (classification)
# Anthropic defaults: claude-sonnet-4-5 (reasoning), claude-haiku-4-5 (classification)
REASONING_MODEL={current.get('REASONING_MODEL', 'gemini-2.5-pro')}
CLASSIFICATION_MODEL={current.get('CLASSIFICATION_MODEL', 'gemini-2.5-flash')}

# Legacy fallback (used if --provider is not set and LLM_PROVIDER is not set)
LLM_API_KEY={current.get('LLM_API_KEY', '')}
LLM_BASE_URL={current.get('LLM_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/')}

# ================================================================
# CallRail API (Script 01)
# ================================================================
CALLRAIL_API_KEY={current.get('CALLRAIL_API_KEY', '')}
CALLRAIL_ACCOUNT_ID={current.get('CALLRAIL_ACCOUNT_ID', '')}

# ================================================================
# SQL Server (Scripts 01, 02, 03, 04, 05)
# ================================================================
SQLSERVER_SERVER={current.get('SQLSERVER_SERVER', '')}
SQLSERVER_DATABASE={current.get('SQLSERVER_DATABASE', '')}
SQLSERVER_UID={current.get('SQLSERVER_UID', '')}
SQLSERVER_PWD={current.get('SQLSERVER_PWD', '')}
SQLSERVER_PORT={current.get('SQLSERVER_PORT', '1433')}

# ================================================================
# Google Sheets (Script 02)
# ================================================================
SPREADSHEET_ID={current.get('SPREADSHEET_ID', '')}
"""

# Write to .env.new for review
new_path = env_path + ".new"
with open(new_path, "w") as f:
    f.write(new_env)

print(f"Written to: {new_path}")
print(f"Review it, then run:")
print(f"  mv Scripts/.env Scripts/.env.bak && mv Scripts/.env.new Scripts/.env")
