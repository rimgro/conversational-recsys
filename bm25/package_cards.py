"""Package an already trained notebook index; never rebuild or retrain it."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from cards_model import load_cards

FILES = ['params.index.json', 'vocab.index.json', 'indices.csc.index.npy',
         'data.csc.index.npy', 'indptr.csc.index.npy', 'items.parquet', 'tokenization.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'indexes' / 'cards'
        target.mkdir(parents=True)
        hashes = {}
        for name in FILES:
            shutil.copyfile(args.source / name, target / name)
            h = hashlib.sha256()
            with (target / name).open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            hashes[name] = h.hexdigest()
        version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
        meta = {'format': 'bm25s-cards-v1', 'library_version': '0.3.13',
                'index_version': version, 'sha256': hashes}
        (target / 'meta.json').write_text(json.dumps(meta, indent=2))
        model = load_cards(target)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, 'x', compression=ZIP_DEFLATED) as archive:
            for f in sorted(target.iterdir()):
                archive.write(f, f.relative_to(Path(tmp)))
        print(f"Packaged {len(model['item_ids'])} tracks; index_version={version}")
        print(args.output.resolve())


if __name__ == '__main__':
    main()
