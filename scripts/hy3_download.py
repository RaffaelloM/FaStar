#!/usr/bin/env python3
"""Resumable single-copy downloader for the HY3 GGUF (~145 GB).

Writes directly to OUT (one copy, no HF cache duplication) and resumes via HTTP
Range from the current file size. Robust to dropped connections: the outer loop
re-issues a Range request from the current offset. Periodically logs progress.
"""
import os, sys, time, requests

URL = "https://huggingface.co/satgeze/Hy3-1M-GGUF/resolve/main/hy3-1M-MTP-Q3_K_M.gguf"
OUT = "/home/raffaele/Progetti/FaStar/hy3-1M-MTP-Q3_K_M.gguf"
CHUNK = 8 * 1024 * 1024
LOG_EVERY = 2 * 1024 * 1024 * 1024  # log a line every ~2 GB


def total_size():
    h = requests.head(URL, allow_redirects=True, timeout=60)
    h.raise_for_status()
    return int(h.headers["Content-Length"])


def main():
    total = total_size()
    print(f"[dl] total={total} bytes ({total/1e9:.2f} GB)", flush=True)
    start = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"[dl] resuming from {start} bytes ({start/total*100:.2f}%)", flush=True)
    last_log = start
    last_time = time.time()
    while start < total:
        try:
            hdr = {"Range": f"bytes={start}-"}
            r = requests.get(URL, headers=hdr, stream=True, timeout=180, allow_redirects=True)
            if r.status_code not in (200, 206):
                print(f"[dl] unexpected status {r.status_code}; retrying in 5s", flush=True)
                time.sleep(5); continue
            r.raise_for_status()
            with open(OUT, "ab") as f:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    start += len(chunk)
                    if start - last_log >= LOG_EVERY:
                        dt = max(time.time() - last_time, 1e-6)
                        mbps = (start - last_log) / dt / 1e6
                        print(f"[dl] {start/1e9:6.2f}/{total/1e9:.2f} GB "
                              f"({start/total*100:5.1f}%) {mbps:5.1f} MB/s", flush=True)
                        last_log = start
                        last_time = time.time()
        except Exception as e:
            print(f"[dl] error at {start} ({start/total*100:.2f}%): {e!r}; retrying in 5s", flush=True)
            time.sleep(5)
            start = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"[dl] DONE {start} == {total} bytes", flush=True)


if __name__ == "__main__":
    main()