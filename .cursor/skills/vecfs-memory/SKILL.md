---
name: vecfs-memory
description: Gives the agent persistent long-term memory using VecFS. Activates context sweep at task start and reflective learning after task completion.
---

# VecFS Memory Skill

## On every non-trivial task — Context Sweep
1. Extract keywords from the current task prompt
2. Call the VecFS `search` tool with those keywords
3. Incorporate any relevant results into your reasoning before starting

## On every completed task — Reflective Learning
1. Identify key lessons, decisions, patterns, or corrections from the task
2. Filter for long-term value — skip one-time commands and session-specific details
3. Call the VecFS `memorize` tool with a descriptive ID like `task_[topic]_[YYYYMMDD]`

## Feedback Loop
- If recalled context helped → call `feedback` with a positive score
- If recalled context was wrong → call `feedback` with a negative score
- If task completed without correction → record a small positive baseline

## Memory categories worth capturing
- Architecture decisions
- Bug root causes and fixes
- Patterns used in this codebase
- Deployment and environment specifics
- Library and framework choices