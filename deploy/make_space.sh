#!/usr/bin/env bash
# Pack the files the website needs into deploy/hf-space/glassquill-app.tar.gz.
# Then upload the three files in deploy/hf-space (README.md, Dockerfile and the archive)
# to the Space (Files > Add file > Upload files); the Space rebuilds on its own.
#
#   bash deploy/make_space.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# committed files only, so local experiments never end up on the Space
git ls-files -z -- server.py examples.json requirements-server.txt \
    src/__init__.py src/detector.py src/features.py src/explain.py \
    models 'reports/*.csv' web ':!web/node_modules' ':!web/dist' |
  tar --null -T - -czf deploy/hf-space/glassquill-app.tar.gz --owner=0 --group=0

echo "deploy/hf-space/glassquill-app.tar.gz: $(du -h deploy/hf-space/glassquill-app.tar.gz | cut -f1)"
tar -tzf deploy/hf-space/glassquill-app.tar.gz | sed 's/^/  /'
