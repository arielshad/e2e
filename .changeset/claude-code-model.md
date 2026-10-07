---
"e2e": minor
---

`claudeCode()` from `e2e/oauth/claude-code` runs agent steps, assertions, and `e2e explore` on a Claude plan through the Claude Code CLI on your machine: sign in by running `claude`, then set `model: claudeCode('sonnet')`. Each model call runs one `claude -p` process isolated from your Claude Code setup and from a Claude Code session you run `e2e` from, with `ANTHROPIC_API_KEY` left out so the plan is what pays. The agent's tools come back as tool calls `e2e` runs itself, so secrets, budgets, and recording work as with any other model. Options: `maxConcurrent` caps the CLI processes across all workers, `effort` sets `--effort`, `env` adds variables, and `executable` names the CLI. `opus[1m]` selects the 1M-token context.
