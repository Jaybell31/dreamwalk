import json
import os
import sys
import time
import urllib.error
import urllib.request

RELAY_URL_SOURCE = (
    # BUGFIX by operator (was: .../main/RELAY.txt -> 404; repo branch is
    # 'master'. Our HARNESS_MODE.md doc said 'main' — doc fixed too.)
    "https://raw.githubusercontent.com/Jaybell31/dreamwalk/master/RELAY.txt"
)
EXPECTED_VERSION = "dreamwalk-selfserve-v2"
SUBMISSION_FILE = "submission.json"


def fetch_with_retry(url, max_retries=3, delay=2):
  for attempt in range(max_retries):
    try:
      req = urllib.request.Request(
          url, headers={"User-Agent": "DreamwalkCourier/1.0"}
      )
      with urllib.request.urlopen(req, timeout=10) as response:
        return response.read().decode("utf-8")
    except Exception as e:
      print(f"Attempt {attempt+1} failed for {url}: {e}")
      if attempt == max_retries - 1:
        raise
      time.sleep(delay * (2**attempt))


def post_with_retry(url, data_dict, max_retries=3, delay=2):
  data_bytes = json.dumps(data_dict).encode("utf-8")
  for attempt in range(max_retries):
    try:
      req = urllib.request.Request(
          url,
          data=data_bytes,
          headers={
              "Content-Type": "application/json",
              "User-Agent": "DreamwalkCourier/1.0",
          },
          method="POST",
      )
      with urllib.request.urlopen(req, timeout=15) as response:
        return response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
      error_body = e.read().decode("utf-8") if e.fp else ""
      print(
          f"HTTP Error {e.code} on attempt {attempt+1}:"
          f" {e.reason}\n{error_body}"
      )
      if attempt == max_retries - 1:
        raise
      time.sleep(delay * (2**attempt))
    except Exception as e:
      print(f"POST Attempt {attempt+1} failed: {e}")
      if attempt == max_retries - 1:
        raise
      time.sleep(delay * (2**attempt))


def validate_and_cap(sub):
  # Enforce strict hard caps client-side to prevent server bounce
  caps = {
      "title": 100,
      "content": 2000,
      "mechanism": 1000,
      "test": 500,
      "visitor": 100,
  }
  for field, limit in caps.items():
    if field in sub and sub[field]:
      if len(sub[field]) > limit:
        print(
            f"Warning: Field '{field}' exceeds limit ({len(sub[field])} >"
            f" {limit}). Truncating."
        )
        sub[field] = sub[field][:limit]
  return sub


def main():
  print("Fetching live base URL from RELAY.txt...")
  relay_text = fetch_with_retry(RELAY_URL_SOURCE)
  # BUGFIX by operator: RELAY.txt line 1 is a '#' comment; take the first
  # line that starts with https:// (Gem's original took splitlines()[0]).
  base_url = ""
  for line in relay_text.strip().splitlines():
    line = line.strip()
    if line.startswith("https://"):
      base_url = line
      break
  if not base_url:
    print(f"ERROR: no https:// line found in RELAY.txt:\n{relay_text}")
    sys.exit(1)
  print(f"Base URL discovered: {base_url}")

  print("Checking /version...")
  version_resp = fetch_with_retry(f"{base_url}/version")
  if EXPECTED_VERSION not in version_resp:
    print(
        f"ERROR: Expected version tag '{EXPECTED_VERSION}' not found in"
        f" response: {version_resp}"
    )
    sys.exit(1)
  print(f"Version verified: {version_resp.strip()}")

  print("Fetching /walk...")
  walk_resp = fetch_with_retry(f"{base_url}/walk")
  print(f"Walk fetched successfully (length: {len(walk_resp)} chars).")

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
