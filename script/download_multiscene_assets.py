"""Direct, resumable downloader for the official Gaussian Grouping assets."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time

FILES = {
    "figurines_data.zip": ("data/lerf_mask/figurines.zip", 412520786, "83eae6c65e5a0041632340f39230acc09ef7714dc0de9d7f352462707f54ef6b"),
    "figurines_checkpoint.zip": ("checkpoint/lerf_mask/figurines.zip", 948325901, "3a7bf91cad7f189c4745ce21bb88b4bc8bcf65b7b5282a59afb1f847592f2ac8"),
    "teatime_data.zip": ("data/lerf_mask/teatime.zip", 257081615, "8e1ac21925d0eb5f4a0a41a71404f73a78f1dcd84bdc046049b6b3e723952ca1"),
    "teatime_checkpoint.zip": ("checkpoint/lerf_mask/teatime.zip", 751032921, "d93f1f907197a8cfb807c7b44c180d787eb0fedb1361837fa239de1bde4fa67e"),
}
BASE = "https://huggingface.co/mqye/Gaussian-Grouping/resolve/main/{}?download=true"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def fetch(url, start, end, part, expected, attempts=20):
    if os.path.isfile(part) and os.path.getsize(part) == expected: return
    for attempt in range(attempts):
        try:
            completed = subprocess.run(["curl.exe", "--noproxy", "*", "-L", "--fail", "--silent", "--show-error",
                                        "--connect-timeout", "30", "--max-time", "180", "--range",
                                        "{}-{}".format(start, end), "-o", part + ".tmp", url], check=False)
            if completed.returncode: raise RuntimeError("curl returned {}".format(completed.returncode))
            if os.path.getsize(part + ".tmp") != expected: raise RuntimeError("short range")
            os.replace(part + ".tmp", part); return
        except Exception:
            if os.path.exists(part + ".tmp"): os.remove(part + ".tmp")
            if attempt + 1 == attempts: raise
            time.sleep(min(2 ** min(attempt, 5), 30))


def download(root, name, remote, size, expected_sha, workers, chunk_size):
    target = os.path.join(root, name)
    if os.path.isfile(target) and os.path.getsize(target) == size and digest(target) == expected_sha:
        print("[verified] " + name, flush=True); return
    parts = target + ".parts"; os.makedirs(parts, exist_ok=True)
    jobs = []
    for index, start in enumerate(range(0, size, chunk_size)):
        end = min(size - 1, start + chunk_size - 1)
        jobs.append((BASE.format(remote), start, end, os.path.join(parts, "{:05d}".format(index)), end-start+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, *job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % max(1, len(jobs)//20) == 0 or done == len(jobs):
                print("[download] {} {:.1%}".format(name, done/len(jobs)), flush=True)
    with open(target + ".assembling", "wb") as output:
        for _, _, _, part, _ in jobs:
            with open(part, "rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""): output.write(block)
    os.replace(target + ".assembling", target)
    actual = digest(target)
    if actual != expected_sha: raise RuntimeError("{} SHA256 mismatch: {}".format(name, actual))
    with open(target + ".sha256.json", "w") as stream:
        json.dump({"sha256": actual, "size": size, "source": BASE.format(remote), "proxy_disabled": True}, stream, indent=2)
    print("[verified] {} {}".format(name, actual), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--chunk-mb", type=int, default=8)
    args = parser.parse_args(); os.makedirs(args.out, exist_ok=True)
    for name, values in FILES.items(): download(args.out, name, *values, args.workers, args.chunk_mb*1024*1024)


if __name__ == "__main__": main()
