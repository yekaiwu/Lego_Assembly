#!/bin/bash
# Pre-commit hook to prevent committing secrets

echo "🔍 Checking for secrets..."

# Check if detect-secrets is installed
if ! command -v detect-secrets &> /dev/null; then
    echo "⚠️  detect-secrets not installed. Install with: pip install detect-secrets"
    echo "Skipping secrets check..."
    exit 0
fi

# Run detect-secrets scan
if detect-secrets scan --baseline .secrets.baseline $(git diff --cached --name-only); then
    echo "✅ No secrets detected"
    exit 0
else
    echo "❌ Potential secrets detected!"
    echo ""
    echo "To fix this:"
    echo "1. Remove the secret from the file"
    echo "2. Add it to environment variables instead"
    echo "3. If this is a false positive, run: detect-secrets audit .secrets.baseline"
    echo ""
    exit 1
fi
