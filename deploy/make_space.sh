#!/usr/bin/env bash
# Build the site and pack what the Space runs into deploy/hf-space/glassquill-app.tar.gz.
# Then upload the four files in deploy/hf-space (README.md, app.py, requirements.txt and
# the archive) to the Space (Files > Add file > Upload files); it rebuilds on its own.
#
#   bash deploy/make_space.sh
set -euo pipefail
cd "$(dirname "$0")/.."

(cd web && npm ci --no-audit --no-fund && npm run build)

# committed files only (plus the fresh build), so local experiments never reach the Space
{ git ls-files -- server.py examples.json \
    src/__init__.py src/detector.py src/features.py src/explain.py models 'reports/*.csv'
  find web/dist -type f; } |
  tar -T - -czf deploy/hf-space/glassquill-app.tar.gz --owner=0 --group=0

echo "deploy/hf-space/glassquill-app.tar.gz: $(du -h deploy/hf-space/glassquill-app.tar.gz | cut -f1)"
tar -tzf deploy/hf-space/glassquill-app.tar.gz | sed 's/^/  /'
