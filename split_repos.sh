#!/usr/bin/env bash
# ==============================================================================
# CrimeAI — Dual Repository Setup & GitHub Split Helper
# This script helps you split and push your project into two separate GitHub repos:
#   1. Frontend Repo -> Deployed to Vercel (https://your-project.vercel.app)
#   2. Backend Repo  -> Deployed to Render / Railway / Fly.io / GCP / AWS
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "==================================================================="
echo "  🚁 CrimeAI: Two-Repository GitHub Preparation & Deployment Tool  "
echo "==================================================================="
echo ""
echo "This helper prepares two standalone repositories:"
echo "  [1] FRONTEND REPO: Ready for 1-click Vercel deployment"
echo "  [2] BACKEND REPO:  Ready for Docker / Cloud deployment"
echo ""

read -p "Enter your GitHub username or organization (e.g. ranaasad): " GITHUB_USER

if [ -z "$GITHUB_USER" ]; then
    echo "❌ GitHub username cannot be empty. Exiting."
    exit 1
fi

FRONTEND_REPO_NAME="crime-ai-frontend"
BACKEND_REPO_NAME="crime-ai-backend"

echo ""
echo "Target GitHub Repositories:"
echo "  Frontend: https://github.com/$GITHUB_USER/$FRONTEND_REPO_NAME.git"
echo "  Backend:  https://github.com/$GITHUB_USER/$BACKEND_REPO_NAME.git"
echo ""
read -p "Do you want to initialize and push these now? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo ""
    echo "ℹ️  No problem! Here are the manual commands whenever you are ready:"
    echo ""
    echo "--- [METHOD A: PUSH FRONTEND TO GITHUB & VERCEL] ---"
    echo "cd \"$FRONTEND_DIR\""
    echo "git init"
    echo "git add ."
    echo "git commit -m 'Initial commit: React + Vite Frontend for CrimeAI'"
    echo "git branch -M main"
    echo "git remote add origin https://github.com/$GITHUB_USER/$FRONTEND_REPO_NAME.git"
    echo "git push -u origin main"
    echo ""
    echo "--- [METHOD B: PUSH BACKEND TO GITHUB] ---"
    echo "cd \"$PROJECT_ROOT\""
    echo "# If not already a git repository:"
    echo "git init"
    echo "git add . ':!frontend/node_modules' ':!frontend/dist'"
    echo "git commit -m 'Initial commit: CrimeAI Backend & ML Inference Engine'"
    echo "git branch -M main"
    echo "git remote add origin https://github.com/$GITHUB_USER/$BACKEND_REPO_NAME.git"
    echo "git push -u origin main"
    echo ""
    exit 0
fi

# 1. Prepare Frontend Repo
echo ""
echo "🚀 [1/2] Preparing Frontend Repository in $FRONTEND_DIR..."
cd "$FRONTEND_DIR"
if [ ! -d ".git" ]; then
    git init
fi
git add .
git commit -m "feat: React 18 + Vite frontend with Vercel deployment & CI/CD" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/$FRONTEND_REPO_NAME.git"

echo "Attempting to push frontend to https://github.com/$GITHUB_USER/$FRONTEND_REPO_NAME.git ..."
git push -u origin main || echo "⚠️ Push failed. Please ensure the repository exists on GitHub, then run: cd frontend && git push -u origin main"

# 2. Prepare Backend Repo
echo ""
echo "🚀 [2/2] Preparing Backend Repository..."
cd "$PROJECT_ROOT"
if [ ! -d ".git" ]; then
    git init
fi
git add . ':!frontend/node_modules'
git commit -m "feat: CrimeAI Python backend with Dockerfile, CORS, and CI/CD" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/$BACKEND_REPO_NAME.git"

echo "Attempting to push backend to https://github.com/$GITHUB_USER/$BACKEND_REPO_NAME.git ..."
git push -u origin main || echo "⚠️ Push failed. Please ensure the repository exists on GitHub, then run: git push -u origin main"

echo ""
echo "==================================================================="
echo "  ✅ Repositories configured!"
echo "  - Connect $FRONTEND_REPO_NAME to Vercel at https://vercel.com"
echo "  - Set Vercel Environment Variable: VITE_API_URL=<your-backend-url>"
echo "==================================================================="
