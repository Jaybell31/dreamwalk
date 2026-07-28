"""Locate the 1063 composer textarea + 'Post comment' button by OCR anchor.
Coordinates only -- clicks nothing, types nothing.

Per skill law: vision hallucinates coordinates; OCR label-anchoring wins.
"""
import sys, time, subprocess, csv
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

HWND = 983870
m = MeatSuitClient()
m.focus_attach(HWND); time.sleep(0.6)
t = (m.active_window().get("title") or "")
if "1063" not in t:
    print("ASSERT FAILED:", t[:70]); sys.exit(4)

m.keys("END"); time.sleep(2.0)
shot = "/tmp/erd_geom.png"
m.screenshot(shot)
subprocess.run(["tesseract", shot, "/tmp/erd_geom", "tsv"], capture_output=True)

rows = list(csv.DictReader(open("/tmp/erd_geom.tsv", encoding="utf-8",
                                errors="replace"), delimiter="\t"))
words = []
for r in rows:
    try:
        txt = (r.get("text") or "").strip()
        if not txt or float(r.get("conf", -1)) < 40:
            continue
        L, T, W, H = int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"])
        words.append((txt, L, T, W, H, L + W // 2, T + H // 2))
    except Exception:
        continue

def find(*needles):
    out = []
    for w in words:
        low = w[0].lower().strip(".,:")
        if any(n == low for n in needles):
            out.append(w)
    return out

for label in ("post", "comment", "your", "preview", "submit"):
    for w in find(label):
        print("%-9s cx=%-5s cy=%-5s  (L=%s T=%s W=%s H=%s)" % (w[0], w[5], w[6],
                                                              w[1], w[2], w[3], w[4]))
print("--- screen ---", m.screen_info())
