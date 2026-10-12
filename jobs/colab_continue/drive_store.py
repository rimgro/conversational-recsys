"""Durable artifact uploads to a selected private Drive folder."""
from pathlib import Path
import hashlib
import json
import mimetypes
import time
from urllib.parse import urlparse

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://www.googleapis.com/drive/v3/files"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"


def authorized_session(credentials):
    client = AuthorizedSession(credentials)
    client.headers["Connection"] = "close"
    client.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
        allowed_methods=frozenset(["GET", "HEAD"]), status_forcelist=[429, 500, 502, 503, 504])))
    return client


class DriveStore:
    def __init__(self, token_path, folder_id, quota_project=None):
        credentials = Credentials.from_authorized_user_file(str(token_path),
            scopes=["https://www.googleapis.com/auth/drive.file"])
        if quota_project:
            credentials = credentials.with_quota_project(quota_project)
        self.client = authorized_session(credentials)
        self.folder_id = folder_id
        self.folders = {".": folder_id}
        self.files = {}
        self.uploaded = {}

    @staticmethod
    def checked(response):
        if not response.ok:
            raise RuntimeError(f"Drive request failed: HTTP {response.status_code}")
        return response.json() if response.content else {}

    def lookup(self, parent, name):
        escape = lambda x: x.replace("\\", "\\\\").replace("'", "\\'")
        response = self.client.get(API, params={"q":
            f"'{escape(parent)}' in parents and name='{escape(name)}' and trashed=false",
            "fields": "files(id,name,mimeType,size,md5Checksum)", "pageSize": 100}, timeout=45)
        files = self.checked(response)["files"]
        if len(files) > 1:
            raise RuntimeError("Ambiguous duplicate artifact path: " + name)
        return files[0] if files else None

    def ensure(self, parent, name, folder=False):
        found = self.lookup(parent, name)
        if found:
            if folder and found["mimeType"] != "application/vnd.google-apps.folder":
                raise RuntimeError("Artifact folder conflicts with an existing file")
            return found["id"]
        metadata = {"name": name, "parents": [parent]}
        if folder:
            metadata["mimeType"] = "application/vnd.google-apps.folder"
        # Never replay an ambiguous creation; recover by the exact parent/name.
        try:
            return self.checked(self.client.post(API, params={"fields": "id"},
                json=metadata, timeout=45))["id"]
        except Exception:
            found = self.lookup(parent, name)
            if found:
                return found["id"]
            raise

    def folder(self, relative):
        relative = Path(relative)
        key = str(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact path")
        if key not in self.folders:
            self.folders[key] = self.ensure(self.folder(relative.parent), relative.name, True)
        return self.folders[key]

    def put(self, source, relative):
        source, relative = Path(source), Path(relative)
        stamp = (source.stat().st_size, source.stat().st_mtime_ns)
        key = str(relative)
        if self.uploaded.get(key) == stamp:
            return
        parent = self.folder(relative.parent)
        file_id = self.files.get(key) or self.ensure(parent, relative.name)
        self.files[key] = file_id
        size = source.stat().st_size
        response = self.client.patch(UPLOAD + "/" + file_id,
            params={"uploadType": "resumable", "fields": "id,size,md5Checksum"}, json={},
            headers={"X-Upload-Content-Length": str(size),
                "X-Upload-Content-Type": mimetypes.guess_type(source.name)[0] or "application/octet-stream"},
            timeout=45)
        self.checked(response)
        location = response.headers["Location"]
        parsed = urlparse(location)
        if parsed.scheme != "https" or parsed.hostname != "www.googleapis.com":
            raise RuntimeError("Unexpected Drive upload destination")
        offset, failures, digest = 0, 0, hashlib.md5()
        with source.open("rb") as stream:
            while offset < size:
                stream.seek(offset)
                block = stream.read(8 * 1024**2)
                try:
                    result = self.client.put(location, data=block, headers={
                        "Content-Length": str(len(block)),
                        "Content-Range": f"bytes {offset}-{offset+len(block)-1}/{size}"}, timeout=120)
                    if result.status_code == 308:
                        next_offset = int(result.headers.get("Range", "bytes=0--1").split("-")[-1]) + 1
                        if next_offset != offset + len(block):
                            raise RuntimeError("Unexpected resumable-upload offset")
                    else:
                        data = self.checked(result)
                        if int(data["size"]) != size:
                            raise RuntimeError("Incomplete Drive upload")
                    offset += len(block)
                    failures = 0
                except Exception:
                    failures += 1
                    if failures > 3:
                        raise
                    time.sleep(failures)
                    # Query accepted bytes before resending a possibly committed chunk.
                    probe = self.client.put(location, data=b"", headers={
                        "Content-Length": "0", "Content-Range": f"bytes */{size}"}, timeout=45)
                    if probe.status_code == 308:
                        offset = int(probe.headers.get("Range", "bytes=0--1").split("-")[-1]) + 1
                    elif probe.ok:
                        offset = size
                    else:
                        self.checked(probe)
        # Verify the completed object, including silent corruption on transfer.
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        metadata = self.checked(self.client.get(API + "/" + file_id,
            params={"fields": "id,size,md5Checksum"}, timeout=45))
        if int(metadata["size"]) != size or metadata.get("md5Checksum") != digest.hexdigest():
            raise RuntimeError("Drive artifact checksum mismatch: " + key)
        self.uploaded[key] = stamp

    def push(self, root, relatives):
        for rel in relatives:
            self.put(Path(root) / rel, rel)
