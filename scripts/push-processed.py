#!/usr/bin/env python3
"""
Pack data/processed/ (images + CSVs) into a compressed archive and upload to server.
A matching extract script on the server side handles extraction.

Usage:
    python scripts/push-processed.py
"""

import os
import sys
import tarfile
import tempfile
import time

try:
    import paramiko
except ImportError:
    print("Error: paramiko not installed. Run: pip install paramiko")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
HOST       = "192.168.37.90"
USER       = "ferencz.peter"
PASSWORD   = "ferenczpeter1234"
REMOTE_DIR = "/home/ferencz.peter/handwritten-text-recognition-trocr"
ARCHIVE_NAME = "processed.tar.gz"

LOCAL_PROCESSED = "data/processed"

# Only include these — skip verification images and result files
INCLUDE = [
    "images",
    "metadata.csv",
    "train.csv",
    "val.csv",
    "test.csv",
]
# ─────────────────────────────────────────────────────────────────────────────


def check_local_files():
    missing = []
    for item in INCLUDE:
        path = os.path.join(LOCAL_PROCESSED, item)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("Missing files/folders:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)


def create_archive(tmp_path):
    print("Compressing data/processed/ ...")
    file_count = 0

    with tarfile.open(tmp_path, "w:gz", compresslevel=6) as tar:
        for item in INCLUDE:
            path = os.path.join(LOCAL_PROCESSED, item)
            arcname = os.path.join("data/processed", item)
            tar.add(path, arcname=arcname)
            if os.path.isdir(path):
                n = sum(len(files) for _, _, files in os.walk(path))
                print(f"  + {arcname}/  ({n} files)")
                file_count += n
            else:
                print(f"  + {arcname}")
                file_count += 1

    size_mb = os.path.getsize(tmp_path) / 1024 / 1024
    print(f"\nArchive: {size_mb:.1f} MB  ({file_count} files packed)\n")


def upload(tmp_path):
    print(f"Connecting to {USER}@{HOST} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)

    sftp = client.open_sftp()
    remote_path = f"{REMOTE_DIR}/{ARCHIVE_NAME}"

    total = os.path.getsize(tmp_path)
    uploaded = [0]
    start = time.time()

    def progress(sent, total):
        uploaded[0] = sent
        pct = sent / total * 100
        elapsed = time.time() - start
        speed = sent / 1024 / 1024 / elapsed if elapsed > 0 else 0
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.1f}%  {speed:.1f} MB/s", end="", flush=True)

    print(f"Uploading {ARCHIVE_NAME} ({total/1024/1024:.1f} MB) ...")
    sftp.put(tmp_path, remote_path, callback=progress)
    print(f"\n\nUpload complete in {time.time()-start:.1f}s")

    sftp.close()
    client.close()


def main():
    check_local_files()

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_archive(tmp_path)
        upload(tmp_path)
    finally:
        os.remove(tmp_path)

    print(f"\nDone. Run on server to extract:")
    print(f"  bash scripts/extract-processed.sh")


if __name__ == "__main__":
    main()
