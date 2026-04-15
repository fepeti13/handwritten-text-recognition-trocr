#!/usr/bin/env python3
"""
Fetch Slurm log files from the training server.

Usage:
    python scripts/fetch-logs.py              # fetch only new/updated logs
    python scripts/fetch-logs.py --all        # force re-download all logs
    python scripts/fetch-logs.py --list       # just list remote logs, don't download
"""

import argparse
import os
import sys

try:
    import paramiko
except ImportError:
    print("Error: paramiko is not installed. Run: pip install paramiko")
    sys.exit(1)

# ── Server config ─────────────────────────────────────────────────────────────
HOST     = "192.168.37.90"
USER     = "ferencz.peter"
PASSWORD = "ferenczpeter1234"

REMOTE_LOGS_DIR = "/home/ferencz.peter/handwritten-text-recognition-trocr/logs"
LOCAL_LOGS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_EXTENSIONS  = (".out", ".err")     # file extensions to fetch
# ─────────────────────────────────────────────────────────────────────────────


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    return client


def list_remote_logs(sftp):
    all_files = sftp.listdir_attr(REMOTE_LOGS_DIR)
    logs = [
        f for f in all_files
        if f.filename.startswith("slurm-") and f.filename.endswith(LOG_EXTENSIONS)
    ]
    logs.sort(key=lambda f: f.filename)
    return logs


def fetch_logs(force_all=False):
    os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)

    print(f"Connecting to {USER}@{HOST} ...")
    client = connect()
    sftp = client.open_sftp()

    remote_logs = list_remote_logs(sftp)

    if not remote_logs:
        print("No slurm-*.out / slurm-*.err files found on the server.")
        sftp.close()
        client.close()
        return

    print(f"Found {len(remote_logs)} log file(s) on server.\n")

    downloaded = 0
    skipped    = 0

    for remote_file in remote_logs:
        fname       = remote_file.filename
        remote_path = f"{REMOTE_LOGS_DIR}/{fname}"
        local_path  = os.path.join(LOCAL_LOGS_DIR, fname)

        remote_size  = remote_file.st_size
        remote_mtime = remote_file.st_mtime

        # Skip if local copy is already identical (same size + mtime)
        if not force_all and os.path.exists(local_path):
            local_stat = os.stat(local_path)
            if local_stat.st_size == remote_size and int(local_stat.st_mtime) == int(remote_mtime):
                print(f"  [skip]     {fname}  ({remote_size} bytes, up-to-date)")
                skipped += 1
                continue

        sftp.get(remote_path, local_path)
        # Preserve remote modification time
        os.utime(local_path, (remote_mtime, remote_mtime))

        print(f"  [fetched]  {fname}  ({remote_size} bytes)")
        downloaded += 1

    sftp.close()
    client.close()

    print(f"\nDone — {downloaded} downloaded, {skipped} already up-to-date.")
    print(f"Logs saved to: {LOCAL_LOGS_DIR}")


def list_only():
    print(f"Connecting to {USER}@{HOST} ...")
    client = connect()
    sftp = client.open_sftp()

    remote_logs = list_remote_logs(sftp)

    if not remote_logs:
        print("No slurm-*.out / slurm-*.err files found on the server.")
    else:
        print(f"{'File':<25} {'Size':>10}  {'Modified'}")
        print("-" * 55)
        for f in remote_logs:
            from datetime import datetime
            mtime = datetime.fromtimestamp(f.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {f.filename:<23} {f.st_size:>10} bytes  {mtime}")

    sftp.close()
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Slurm logs from training server.")
    parser.add_argument("--all",  action="store_true", help="Force re-download all logs")
    parser.add_argument("--list", action="store_true", help="List remote logs without downloading")
    args = parser.parse_args()

    if args.list:
        list_only()
    else:
        fetch_logs(force_all=args.all)
