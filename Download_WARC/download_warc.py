import os
import sys
import requests
import time
from langdetect import detect
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor

# Ensure script gets at least one WARC file
if len(sys.argv) < 2:
    print("Error: No WARC files provided.", flush=True)
    sys.exit(1)

# Extract assigned WARC files
warc_files = [line.strip() for line in sys.argv[1:]]

# File paths
warc_paths_file = "/ibex/user/abuhanjt/test/CC-MAIN-2024-42/warc.paths"
download_dir = "/ibex/user/abuhanjt/test/CC-MAIN-2024-42/downloaded_warc_files"
processed_warc_list = "/ibex/user/abuhanjt/test/CC-MAIN-2024-42/processed_warc_files.txt"
lock_file = "/ibex/user/abuhanjt/test/CC-MAIN-2024-42/processed_warc.lock"

# Load processed files to avoid re-processing
processed_files = set()
if os.path.exists(processed_warc_list):
    with open(processed_warc_list, "r") as f:
        processed_files = set(line.split(" - ")[0].strip() for line in f)

def write_to_processed_list(filename, status):
    """ Save only the filename using a lock to prevent conflicts """
    filename_only = os.path.basename(filename)
    with FileLock(lock_file):
        with open(processed_warc_list, "a") as f:
            f.write(f"{filename_only} - {status}\n")

def download_warc_file(warc_filename):
    """Download WARC file using multi-threaded chunked downloading."""
    warc_url = f"https://data.commoncrawl.org/{warc_filename}"
    warc_filepath = os.path.join(download_dir, os.path.basename(warc_filename))

    if warc_filename in processed_files:
        print(f"Skipping {warc_filename} (Already processed)")
        return

    print(f"Checking Arabic content in {warc_url}...")

    try:
        response = requests.get(warc_url, headers={"Range": "bytes=0-1024000"}, stream=True, timeout=30)
        if response.status_code == 206:
            content = response.content.decode("utf-8", errors="ignore")
            for line in content.splitlines():
                try:
                    if len(line.strip()) > 20 and detect(line.strip()) == "ar":
                        print(f"Arabic detected in {warc_url}")

                        # Multi-threaded download using chunks
                        CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per chunk
                        file_size = int(requests.head(warc_url).headers.get("Content-Length", 0))
                        chunk_ranges = [(i, min(i + CHUNK_SIZE - 1, file_size - 1)) for i in range(0, file_size, CHUNK_SIZE)]

                        def download_chunk(start, end):
                            headers = {"Range": f"bytes={start}-{end}"}
                            chunk_resp = requests.get(warc_url, headers=headers, stream=True, timeout=30)
                            return chunk_resp.content

                        with open(warc_filepath, "wb") as warc_file:
                            with ThreadPoolExecutor(max_workers=8) as chunk_executor:
                                for chunk in chunk_executor.map(lambda r: download_chunk(*r), chunk_ranges):
                                    warc_file.write(chunk)

                        write_to_processed_list(warc_filename, "Downloaded")
                        return
                except Exception:
                    continue
    except requests.exceptions.RequestException as e:
        print(f"Error processing {warc_url}: {e}")

    print(f"No Arabic content found in {warc_url}, skipping download.")
    write_to_processed_list(warc_filename, "Skipped (No Arabic)")

# Parallel execution
with ThreadPoolExecutor(max_workers=32) as executor:
    executor.map(download_warc_file, warc_files)

print(f"Job finished processing {len(warc_files)} files.")