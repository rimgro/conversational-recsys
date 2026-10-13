"""Create matched left/right serving directories from a verified training export."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from recsys.serving.artifacts import split_bundle
    result = split_bundle(args.bundle, args.output)
    manifest = json.loads((result / "right/serving_manifest.json").read_text())
    print(json.dumps({"output": str(result.resolve()), "bundle_id": manifest["bundle_id"],
        "release_id": manifest["release_id"]}, indent=2))


if __name__ == "__main__":
    main()
