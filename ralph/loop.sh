#!/bin/bash
# Ralph Loop Runner for OpenProse
# Usage: ./loop.sh [plan|build|slc] [max_iterations]

set -euo pipefail

MODE="${1:-build}"
MAX_ITERATIONS="${2:-0}"
ITERATION=0

case "$MODE" in
  plan)
    PROSE_FILE="plan.prose"
    echo "━━━ PLANNING MODE ━━━"
    ;;
  slc)
    PROSE_FILE="plan-slc.prose"
    echo "━━━ SLC PLANNING MODE ━━━"
    ;;
  build)
    PROSE_FILE="build.prose"
    echo "━━━ BUILDING MODE ━━━"
    ;;
  requirements)
    # Requirements is interactive, run once
    echo "━━━ REQUIREMENTS MODE ━━━"
    cat requirements.prose | claude -p --dangerously-skip-permissions
    exit 0
    ;;
  *)
    echo "Usage: ./loop.sh [plan|build|slc|requirements] [max_iterations]"
    exit 1
    ;;
esac

echo "Prose: $PROSE_FILE"
echo "Branch: $(git branch --show-current)"
[ "$MAX_ITERATIONS" -gt 0 ] && echo "Max: $MAX_ITERATIONS iterations"
echo "━━━━━━━━━━━━━━━━━━━━━━"

while true; do
  if [ "$MAX_ITERATIONS" -gt 0 ] && [ "$ITERATION" -ge "$MAX_ITERATIONS" ]; then
    echo "Reached max iterations: $MAX_ITERATIONS"
    break
  fi

  # Run prose file through Claude
  # The prose VM executes the program structure
  cat "$PROSE_FILE" | claude -p \
    --dangerously-skip-permissions \
    --model opus

  # Push after each iteration
  CURRENT_BRANCH=$(git branch --show-current)
  git push origin "$CURRENT_BRANCH" 2>/dev/null || \
    git push -u origin "$CURRENT_BRANCH" 2>/dev/null || true

  ITERATION=$((ITERATION + 1))
  echo -e "\n══════════ LOOP $ITERATION ══════════\n"
done
