import json
import os
import sys
import time
import urllib.error
import urllib.request

# PATCH: Hardcoded to 'master' branch to prevent 404s
RELAY_URL_SOURCE = "https://raw.githubusercontent.com/Jaybell31/dreamwalk/master/RELAY.txt"
EXPECTED_VERSION = "dreamwalk-selfserve-v2"
SUBMISSION_FILE = "submission.json"

def fetch_with_retry(url, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DreamwalkCourier/2.1"})
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # PATCH: Bail immediately on 4xx (except 429 Rate Limit) - don't retry dead ends
            if 400 <= e.code < 500 and e.code != 429:
                print(f"FATAL Client Error {e.code} for {url}: {e.reason}")
                sys.exit(1)
            print(f"HTTP Error {e.code} on attempt {attempt+1}: {e.reason}")
            if attempt == max_retries - 1: raise
            time.sleep(delay * (2**attempt))
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt == max_retries - 1: raise
            time.sleep(delay * (2**attempt))

def post_with_retry(url, data_dict, max_retries=3, delay=2):
    data_bytes = json.dumps(data_dict).encode("utf-8")
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=data_bytes,
                headers={"Content-Type": "application/json", "User-Agent": "DreamwalkCourier/2.1"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            if 400 <= e.code < 500 and e.code != 429:
                print(f"FATAL POST Error {e.code}: {e.reason}\n{error_body}")
                sys.exit(1)
            print(f"HTTP Error {e.code} on attempt {attempt+1}: {e.reason}\n{error_body}")
            if attempt == max_retries - 1: raise
            time.sleep(delay * (2**attempt))
        except Exception as e:
            print(f"POST Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1: raise
            time.sleep(delay * (2**attempt))

def validate_and_cap(sub):
    caps = {"title": 100, "content": 2000, "mechanism": 1000, "test": 500, "visitor": 100}
    for field, limit in caps.items():
        if field in sub and sub[field]:
            if len(sub[field]) > limit:
                print(f"Warning: '{field}' exceeds {limit} chars. Truncating.")
                sub[field] = sub[field][:limit]
    return sub

def main():
    print("Fetching live base URL from RELAY.txt...")
    relay_text = fetch_with_retry(RELAY_URL_SOURCE)

    # PATCH: Defensively scan for the actual URL, ignoring markdown comments
    base_url = None
    for line in relay_text.splitlines():
        line = line.strip()
        if line.startswith("https://"):
            base_url = line
            break

    if not base_url:
        print("ERROR: Could not find a valid https:// URL in RELAY.txt")
        sys.exit(1)

    print(f"Base URL discovered: {base_url}")

    print("Checking /version...")
    version_resp = fetch_with_retry(f"{base_url}/version")
    if EXPECTED_VERSION not in version_resp:
        print(f"ERROR: Expected version '{EXPECTED_VERSION}' not found.")
        sys.exit(1)

    if not os.path.exists(SUBMISSION_FILE):
        print(f"ERROR: Submission file '{SUBMISSION_FILE}' not found.")
        sys.exit(1)

    with open(SUBMISSION_FILE, "r", encoding="utf-8") as f:
        sub = json.load(f)

    sub = validate_and_cap(sub)
    print(f"Posting submission '{sub.get('title')}' to {base_url}/dream...")

    result = post_with_retry(f"{base_url}/dream", sub)
    print(f"Server response:\n{result}")

if __name__ == "__main__":
    main()
