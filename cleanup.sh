#!/usr/bin/env bash
#
# cleanup_tracked_artifacts.sh
#
# Removes files from git tracking that should never have been committed
# (bytecode caches, large MSA artifacts, prediction outputs). It does NOT
# delete your local copies — only `git rm --cached` — so nothing on disk is
# lost. Review the staged deletions, then commit.
#
# Run from the repo root AFTER replacing .gitignore with the new one:
#   cp /path/to/new/.gitignore .
#   bash cleanup_tracked_artifacts.sh
#   git status        # review
#   git commit -m "Stop tracking bytecode, MSA artifacts, and run outputs; add real .gitignore"
#
set -euo pipefail

if [ ! -d .git ]; then
  echo "Error: run this from the repository root (no .git dir found)." >&2
  exit 1
fi

echo "Untracking __pycache__ and compiled Python..."
git rm -r --cached --quiet --ignore-unmatch \
  '**/__pycache__' '__pycache__' '*.pyc' '*.pyo' 2>/dev/null || true

echo "Untracking large MSA / ColabFold test artifacts..."
git rm -r --cached --quiet --ignore-unmatch \
  'scripts/API-Testing/ColabFold-Testing/results' 2>/dev/null || true
git rm --cached --quiet --ignore-unmatch \
  '*.a3m' '*.m8' 'out.tar.gz' 2>/dev/null || true

echo "Untracking prediction run outputs and intermediates..."
git rm --cached --quiet --ignore-unmatch \
  result.pdb myprotein.pdb input.pka benchmark_results.json 2>/dev/null || true

echo
echo "Done. Local files are untouched. Review with 'git status', then commit."
echo "Tip: keep benchmarks/results.jsonl and benchmarks/BENCHMARKS.md tracked —"
echo "they're your reproducible benchmark log."