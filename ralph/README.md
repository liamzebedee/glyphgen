# Ralph Loop - OpenProse Implementation

Minimal OpenProse implementation of [Geoff Huntley's Ralph methodology](https://ghuntley.com/ralph/).

## Three Phases, Two Prompts, One Loop

```
Phase 1: Define Requirements    →  specs/*.md
Phase 2: Planning Mode          →  IMPLEMENTATION_PLAN.md
Phase 3: Building Mode          →  implement → test → commit → loop
```

## Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `requirements.prose` | Interview → JTBD → specs | Starting a new project |
| `plan.prose` | Gap analysis → task list | No plan, or plan is stale |
| `build.prose` | Pick task → implement → commit | Plan exists, ready to build |
| `plan-slc.prose` | SLC release planning | Want coherent product releases |
| `ralph.prose` | Full workflow combined | Reference/one-shot execution |

## Usage

### Option 1: Bash Loop (Classic Ralph)

The outer loop is intentionally dumb:

```bash
# Planning mode
while :; do /prose-run plan.prose ; done

# Building mode
while :; do /prose-run build.prose ; done
```

### Option 2: Run Phases Sequentially

```bash
# 1. Define requirements (interactive)
/prose-run requirements.prose

# 2. Generate plan
/prose-run plan.prose

# 3. Build (run in loop)
while :; do /prose-run build.prose ; done
```

## Key Principles

**Context is everything**
- Use main agent as scheduler, subagents for work
- Each loop iteration starts fresh (context cleared)
- State persists via FILES on disk

**Steer upstream** (deterministic setup)
- First ~5K tokens for specs
- Every loop loads same files: PROMPT + AGENTS.md
- Your code patterns shape what gets generated

**Steer downstream** (backpressure)
- Tests, typechecks, lints reject invalid work
- AGENTS.md specifies project-specific commands

**Let Ralph Ralph**
- Don't micromanage task selection
- LLM self-identifies, self-corrects
- Eventual consistency through iteration
- Plan is disposable - regenerate when wrong

## Required Files

Your project needs:

```
project/
├── specs/              # One .md per topic of concern
│   └── *.md
├── AGENTS.md           # How to build/test (operational only)
├── IMPLEMENTATION_PLAN.md  # Prioritized task list (generated)
└── src/
    └── lib/            # Shared utilities (patterns Ralph follows)
```

### AGENTS.md Template

```markdown
## Build & Run
[How to build the project]

## Validation
- Tests: `npm test`
- Typecheck: `npm run typecheck`
- Lint: `npm run lint`

## Operational Notes
[Learnings about running the project]
```

## The Loop Mechanic

```
┌─────────────────────────────────────────────────────┐
│                    OUTER LOOP                        │
│   while :; do /prose-run build.prose ; done         │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                  build.prose                         │
│                                                      │
│  1. ORIENT   - Load specs, agents, patterns         │
│  2. SELECT   - Pick most important task from plan   │
│  3. INVESTIGATE - Search before assuming            │
│  4. IMPLEMENT - Do the work completely              │
│  5. VALIDATE - Run tests (backpressure)             │
│  6. UPDATE   - Mark done, note discoveries          │
│  7. COMMIT   - Persist to git                       │
│                                                      │
│  EXIT → Context cleared → Loop restarts fresh       │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
        IMPLEMENTATION_PLAN.md updated on disk
        (state persists between iterations)
```

## Guardrails (The 9s)

From the build prompt, in order of criticality:

- Capture the WHY in documentation
- Single source of truth, no migrations
- Fix unrelated test failures in same increment
- Create git tags on clean builds
- Keep IMPLEMENTATION_PLAN.md current
- Update AGENTS.md with operational learnings (brief!)
- Resolve or document discovered bugs
- Implement completely - no placeholders
- Clean completed items periodically
