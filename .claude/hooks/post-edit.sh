#!/bin/bash
# After any file edit, check for common mistakes

FILE="$CLAUDE_FILE_PATH"

# Block secrets from being committed
if [ -f "$FILE" ]; then
  if grep -qE '(sk-ant-|AGENT_API_KEY=(?!<your))' "$FILE" 2>/dev/null; then
    echo "BLOCKED: Possible API key detected in $FILE"
    exit 1
  fi
fi