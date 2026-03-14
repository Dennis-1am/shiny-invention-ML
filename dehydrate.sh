#!/bin/bash

echo "syncing requirements.txt..."
# 1. Update the requirements list so we can rebuild later
.venv/bin/pip freeze > requirements.txt

echo "deactivating environment..."
# 2. Stop using the environment
deactivate 2>/dev/null

echo "removing .venv folder to save space..."
# 3. The 'Heavy Lifting' (deleting the folder)
rm -rf .venv

echo "✨ Project dehydrated. Run 'python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' to rehydrate."