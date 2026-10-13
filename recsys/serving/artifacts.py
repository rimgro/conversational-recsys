"""Matched, verified serving subsets of an immutable training export."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

RIGHT_FILES = {"catalog.json", "track_vectors.npy", "tracks.hnsw"}
LEFT_FILES = {"catalog.json", "config.json", "history_features.npz", "query_trainable.pt"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_id(manifest):
    # Includes item/index/history weights as well as the query adapter.
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def safe_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not isinstance(name, str) or Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Unsafe artifact path")
    if not path.is_file():
        raise ValueError("Missing artifact: " + name)
    return path


def validate_source(manifest):
    if (manifest.get("schema_version") != 1 or manifest.get("space") != "cosine"
            or manifest.get("dimension") != 256 or not manifest.get("bundle_id")
            or not isinstance(manifest.get("tracks"), int) or manifest["tracks"] < 1):
        raise ValueError("Unsupported Qwen export")
    hashes = manifest.get("sha256", {})
    needed = RIGHT_FILES | LEFT_FILES
    if not needed.issubset(hashes) or not any(n.startswith("tokenizer/") for n in hashes):
        raise ValueError("Incomplete Qwen export")
    for checksum in hashes.values():
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("Invalid export checksum")


def load_subset(root, role):
    root = Path(root)
    serving = json.loads((root / "serving_manifest.json").read_text())
    source = json.loads((root / "source_manifest.json").read_text())
    validate_source(source)
    if (serving.get("schema_version") != 1 or serving.get("role") != role
            or serving.get("release_id") != release_id(source)
            or serving.get("bundle_id") != source["bundle_id"]
            or serving.get("dimension") != source["dimension"]
            or serving.get("tracks") != source["tracks"]):
        raise ValueError("Serving release identity mismatch")
    expected = RIGHT_FILES if role == "right" else LEFT_FILES | {
        name for name in source["sha256"] if name.startswith("tokenizer/")}
    if set(serving.get("sha256", {})) != expected:
        raise ValueError("Incomplete serving subset")
    for name in expected:
        checksum = serving["sha256"][name]
        if checksum != source["sha256"][name] or sha256(safe_file(root, name)) != checksum:
            raise ValueError("Artifact integrity failure: " + name)
    return serving, source


def split_bundle(source, destination):
    """Verify all source files; atomically create new left/right directories."""
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    validate_source(manifest)
    for name, checksum in manifest["sha256"].items():
        if sha256(safe_file(source, name)) != checksum:
            raise ValueError("Source integrity failure: " + name)
    if destination.exists():
        raise ValueError("Destination exists; use a new release directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".serving-", dir=destination.parent))
    try:
        for role, names in (("right", RIGHT_FILES), ("left", LEFT_FILES | {
                name for name in manifest["sha256"] if name.startswith("tokenizer/")})):
            root = stage / role
            root.mkdir()
            for name in sorted(names):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(safe_file(source, name), target)
            hashes = {name: manifest["sha256"][name] for name in sorted(names)}
            serving = {"schema_version": 1, "role": role, "bundle_id": manifest["bundle_id"],
                "release_id": release_id(manifest), "dimension": 256, "tracks": manifest["tracks"],
                "sha256": hashes}
            (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
            (root / "serving_manifest.json").write_text(json.dumps(serving, indent=2))
            load_subset(root, role)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination
