"""Check that the published core matches the live training snapshot."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/training_code_manifest.json').read_text())
for name, expected in manifest['core_sha256'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Core differs from the training snapshot: ' + name)
print('Verified', len(manifest['core_sha256']), 'core files against the training snapshot')
