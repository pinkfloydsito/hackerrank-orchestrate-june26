import json
import sys
from pathlib import Path

for path in Path('data/raw').rglob('*_annotations.coco.json'):
    with open(path) as f:
        data = json.load(f)
    print(f"=== {path} ===")
    print(f"Categories: {[c['name'] for c in data.get('categories', [])]}")
    print(f"Images: {len(data.get('images', []))}")
    print(f"Annotations: {len(data.get('annotations', []))}")
    print()
