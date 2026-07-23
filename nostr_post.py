#!/usr/bin/env python3
"""nostr_post.py — minimal Nostr client for the Dream Walk lab identity.

Generates/loads a keypair, signs NIP-01 events (BIP-340 Schnorr via
coincurve), and publishes to public relays over websocket.

Usage:
  nostr_post.py whoami                     # npub + hex pubkey
  nostr_post.py post "text"                # kind-1 note to default relays
  nostr_post.py post-file /path/to/file    # file contents as the note
  nostr_post.py profile                    # publish kind-0 profile metadata

Key stored at ~/.dreamwalk_nostr_key (hex, chmod 600). BACK THIS UP —
it IS the lab's identity.
"""

import hashlib
import json
import os
import sys
import time

import coincurve
from websocket import create_connection

KEY_PATH = os.path.expanduser("~/.dreamwalk_nostr_key")

RELAYS = [
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://relay.nostr.band",
    "wss://relay.primal.net",
    "wss://nostr.mom",
    "wss://offchain.pub",
]

PROFILE = {
    "name": "dreamwalk",
    "display_name": "Dream Walk — the novel-idea brain",
    "about": ("Live research exchange for AI and human minds. 19k+ tested idea "
              "fragments, a blind nightly court, real-data experiments, LAWS "
              "that survived, a GRAVEYARD of executed ideas. Any AI can plug "
              "in — free. Nothing epistemic is for sale.\n"
              "https://github.com/Jaybell31/dreamwalk"),
    "website": "https://github.com/Jaybell31/dreamwalk",
    "lud16": "",  # lightning address later — sats patronage
}


# ---- bech32 (BIP-173) for npub display --------------------------------------
B32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _polymod(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if ((b >> i) & 1) else 0
    return chk


def bech32_encode(hrp, data8):
    data5 = []
    acc = bits = 0
    for b in data8:
        acc = (acc << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            data5.append((acc >> bits) & 31)
    if bits:
        data5.append((acc << (5 - bits)) & 31)
    values = [ord(c) & 31 for c in hrp] 
    values = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp] + data5
    poly = _polymod(values + [0] * 6) ^ 1
    checksum = [(poly >> 5 * (5 - i)) & 31 for i in range(6)]
    d5 = []
    acc = bits = 0
    for b in data8:
        acc = (acc << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            d5.append((acc >> bits) & 31)
    if bits:
        d5.append((acc << (5 - bits)) & 31)
    return hrp + "1" + "".join(B32[d] for d in d5 + checksum)


# ---- key management ----------------------------------------------------------

def load_key():
    if os.path.exists(KEY_PATH):
        sk_hex = open(KEY_PATH).read().strip()
    else:
        sk_hex = coincurve.PrivateKey().to_hex()
        with open(KEY_PATH, "w") as f:
            f.write(sk_hex + "\n")
        os.chmod(KEY_PATH, 0o600)
        print("NEW IDENTITY generated -> %s (BACK IT UP)" % KEY_PATH, file=sys.stderr)
    sk = coincurve.PrivateKey(bytes.fromhex(sk_hex))
    pub_xonly = sk.public_key.format(compressed=True)[1:33]
    return sk, pub_xonly.hex()


# ---- NIP-01 event ------------------------------------------------------------

def make_event(sk, pubkey_hex, kind, content, tags=None):
    tags = tags or []
    created = int(time.time())
    payload = [0, pubkey_hex, created, kind, tags, content]
    ser = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    eid = hashlib.sha256(ser.encode("utf-8")).hexdigest()
    sig = sk.sign_schnorr(bytes.fromhex(eid)).hex()
    return {"id": eid, "pubkey": pubkey_hex, "created_at": created,
            "kind": kind, "tags": tags, "content": content, "sig": sig}


def publish(event, relays=RELAYS, timeout=10):
    msg = json.dumps(["EVENT", event], ensure_ascii=False)
    results = {}
    for url in relays:
        try:
            ws = create_connection(url, timeout=timeout)
            ws.send(msg)
            reply = ws.recv()
            ws.close()
            ok = json.loads(reply)
            results[url] = "accepted" if (len(ok) >= 3 and ok[2] is True) else repr(reply)[:120]
        except Exception as e:
            results[url] = "FAIL: %r" % e
    return results


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    sk, pub = load_key()

    if cmd == "whoami":
        print("hex pubkey:", pub)
        print("npub:      ", bech32_encode("npub", bytes.fromhex(pub)))
        return

    if cmd == "profile":
        ev = make_event(sk, pub, 0, json.dumps(PROFILE, ensure_ascii=False))
    elif cmd == "post":
        ev = make_event(sk, pub, 1, sys.argv[2])
    elif cmd == "post-file":
        ev = make_event(sk, pub, 1, open(sys.argv[2]).read().strip())
    else:
        sys.exit("unknown command %r" % cmd)

    res = publish(ev)
    print("event id:", ev["id"])
    for url, status in res.items():
        print("  %-28s %s" % (url, status))
    accepted = sum(1 for s in res.values() if s == "accepted")
    print("accepted by %d/%d relays" % (accepted, len(res)))
    if accepted and ev["kind"] == 1:
        print("view: https://njump.me/%s" % ev["id"])


if __name__ == "__main__":
    main()
