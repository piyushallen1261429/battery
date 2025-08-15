#!/bin/bash
# Simple GitHub sync script
# Usage:
#   ./github_sync.sh pull  -> Pull latest changes from GitHub
#   ./github_sync.sh push  -> Push local changes to GitHub

REPO_URL="git@github.com:piyushallen1261429/battery.git"
BRANCH="main"

if [ "$1" == "pull" ]; then
    echo "Pulling latest changes from $BRANCH..."
    git pull origin "$BRANCH" --allow-unrelated-histories
    exit 0
fi

if [ "$1" == "push" ]; then
    echo "Adding all changes..."
    git add .
    echo "Committing changes..."
    git commit -m "Sync from $(hostname) on $(date)"
    echo "Pushing to $BRANCH..."
    git push origin "$BRANCH" --force
    exit 0
fi

echo "Usage: $0 {pull|push}"
exit 1
