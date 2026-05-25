#!/bin/bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command // ""')

# Extract remote name from command (e.g. "git push origin main" → "origin")
remote=$(echo "$cmd" | sed -nE 's/.*git[[:space:]]+push[[:space:]]+([^[:space:]]+).*/\1/p')
[ -z "$remote" ] && remote="origin"

url=$(git remote get-url "$remote" 2>/dev/null || echo "")

if echo "$url" | grep -qi 'bitbucket'; then
  jq -n --arg reason "Push blocked: remote '$remote' points to Bitbucket ($url). Only GitHub pushes are allowed." \
    '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":$reason}}'
fi
