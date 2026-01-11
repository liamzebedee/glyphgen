# AGENTS.md - Operational Guide

Keep this file brief (~60 lines max). Status updates belong in IMPLEMENTATION_PLAN.md.

## Build & Run

```bash
# Build
npm run build

# Dev server
npm run dev

# Production
npm run start
```

## Validation (Backpressure)

Run these after implementing to get immediate feedback:

```bash
# Tests (specific file)
npm test -- path/to/file.test.ts

# Tests (all)
npm test

# Typecheck
npm run typecheck

# Lint
npm run lint
```

## Codebase Patterns

- Shared utilities live in `src/lib/`
- Prefer consolidated implementations over ad-hoc copies
- Follow existing patterns when adding new code

## Operational Notes

<!-- Add learnings about how to run the project here -->
<!-- Example: "Use --legacy-peer-deps for npm install" -->
