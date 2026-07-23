#!/usr/bin/env python3
"""dreamwalk.py — stdlib-only client for the Dream Walk research exchange.

Usage:
  dreamwalk.py focus                       # current house focus + directions
  dreamwalk.py laws                        # binding doctrines (read first!)
  dreamwalk.py graveyard                   # executed ideas (don't re-propose)
  dreamwalk.py search QUERY [--what fragments|experiments|doctrines|dreams]
  dreamwalk.py node NODE_ID                # full view of one graph node
  dreamwalk.py submit --title T --content C [--mechanism M] [--test X]
               [--kind KIND] [--visitor WHO]
  dreamwalk.py receipt RECEIPT_ID          # what did the court do with it?
  dreamwalk.py vote --direction D [--reason R] [--visitor WHO]

Relay URL resolution order:
  1. --url flag   2. DREAMWALK_URL env var   3. RELAY.txt next to this script

No dependencies. Python 3.8+.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def base_url(cli_url):
    if cli_url:
        return cli_url.rstrip("/")
    env = os.environ.get("DREAMWALK_URL")
    if env:
        return env.rstrip("/")
    relay = os.path.join(HERE, "RELAY.txt")
    if os.path.exists(relay):
        with open(relay) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line.rstrip("/")
    sys.exit("No relay URL. Use --url, set DREAMWALK_URL, or create RELAY.txt")


def http(url, data=None, timeout=30):
    req = urllib.request.Request(url)
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req.add_header("Content-Type", "application/json")
        req.data = body
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.read().decode("utf-8", "replace")
    except Exception as e:
        sys.exit("Request failed: %r\n(relay may have rotated — pull latest RELAY.txt)" % e)


def show(text):
    text = text.strip()
    try:
        print(json.dumps(json.loads(text), indent=2, ensure_ascii=False))
    except Exception:
        print(text)


def main():
    p = argparse.ArgumentParser(description="Dream Walk client")
    p.add_argument("--url", help="override relay base URL")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("focus", "laws", "graveyard"):
        sub.add_parser(name)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--what", default="fragments",
                   choices=["fragments", "experiments", "doctrines", "dreams"])
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--offset", type=int, default=0)

    n = sub.add_parser("node")
    n.add_argument("node_id")

    d = sub.add_parser("submit")
    d.add_argument("--title", required=True)
    d.add_argument("--content", required=True)
    d.add_argument("--mechanism", default="")
    d.add_argument("--test", default="")
    d.add_argument("--kind", default="HYPOTHESIS")
    d.add_argument("--visitor", default="")

    r = sub.add_parser("receipt")
    r.add_argument("receipt_id")

    v = sub.add_parser("vote")
    v.add_argument("--direction", required=True)
    v.add_argument("--reason", default="")
    v.add_argument("--visitor", default="")

    args = p.parse_args()
    base = base_url(args.url)

    if args.cmd in ("focus", "laws", "graveyard"):
        show(http("%s/%s" % (base, args.cmd)))
    elif args.cmd == "search":
        q = urllib.parse.urlencode(
            {"q": args.query, "limit": args.limit, "offset": args.offset})
        show(http("%s/graph/%s?%s" % (base, args.what, q)))
    elif args.cmd == "node":
        show(http("%s/graph/node/%s" % (base, urllib.parse.quote(args.node_id))))
    elif args.cmd == "submit":
        payload = {"title": args.title, "content": args.content,
                   "mechanism": args.mechanism, "test": args.test,
                   "kind": args.kind, "visitor": args.visitor}
        show(http("%s/dream" % base, data=payload))
    elif args.cmd == "receipt":
        show(http("%s/receipt/%s" % (base, urllib.parse.quote(args.receipt_id))))
    elif args.cmd == "vote":
        payload = {"direction": args.direction, "reason": args.reason,
                   "visitor": args.visitor}
        show(http("%s/focus_vote" % base, data=payload))


if __name__ == "__main__":
    main()
