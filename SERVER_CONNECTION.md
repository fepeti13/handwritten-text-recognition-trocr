# Server Connection Guide

## Prerequisites

Make sure WireGuard is installed:
```bash
sudo apt install wireguard
```

---

## Step 1: Connect to VPN

```bash
sudo wg-quick up wg0
```

To disconnect:
```bash
sudo wg-quick down wg0
```

---

## Step 2: SSH into the Server

```bash
ssh ferencz.peter@192.168.37.90
```

- **User:** `ferencz.peter`
- **Password:** `ferenczpeter1234`

---

## Step 3: Fetch Files from the Server (SCP)

### Copy a single file to your Desktop:
```bash
scp ferencz.peter@192.168.37.90:~/handwritten-text-recognition-trocr/data/processed/test_predictions_finetuned.csv ~/Desktop/
```

### Copy all Slurm log files to local `logs/` folder:
```bash
scp ferencz.peter@192.168.37.90:~/handwritten-text-recognition-trocr/logs/slurm-*.out ./logs/
```

### Copy entire logs directory:
```bash
scp -r ferencz.peter@192.168.37.90:~/handwritten-text-recognition-trocr/logs/ ./logs/
```

---

## Project Paths on Server

| What | Path |
|---|---|
| Project root | `~/handwritten-text-recognition-trocr/` |
| Slurm log files | `~/handwritten-text-recognition-trocr/logs/slurm-*.out` |
| Processed data | `~/handwritten-text-recognition-trocr/data/processed/` |
| Trained models | `~/handwritten-text-recognition-trocr/models/` |

---

## Useful Commands on Server

```bash
# Check running/queued jobs
squeue -u ferencz.peter

# Check job status
sacct -u ferencz.peter --format=JobID,JobName,State,Elapsed

# Monitor GPU usage
nvidia-smi

# Watch a log file live
tail -f logs/slurm-<job_id>.out
```

---

## Notes

- The VPN (`wg0`) must be active before SSH/SCP will work.
- Password authentication is used — consider setting up SSH keys for convenience (see below).

### Optional: Set up SSH key (no password needed)
```bash
ssh-keygen -t ed25519 -C "fepeti13@gmail.com"
ssh-copy-id ferencz.peter@192.168.37.90
```
