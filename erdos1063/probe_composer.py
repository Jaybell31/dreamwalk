"""Find the real composer textarea by numpy white-box scan + probe-paste.

Law: PROBE BEFORE YOU PASTE. A probe only PASTES a short sentinel (never Enter,
never the submit button), so sweeping candidates is harmless. Whatever accepts
text is the true coordinate.
"""
import sys, time, subprocess
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

HWND = 983870
SENT = "PROBE_7c1d"
POISON = "CLIP_POISON_9f3a"


def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                          capture_output=True).stdout.decode("utf-8", "replace")


def get_clip():
    return ps("Get-Clipboard -Raw").replace("\r\n", "\n").strip()


m = MeatSuitClient()
m.focus_attach(HWND); time.sleep(0.7)
if "1063" not in (m.active_window().get("title") or ""):
    print("ABORT: not on 1063"); sys.exit(4)
m.keys("END"); time.sleep(2.0)

shot = "/tmp/erd_scan.png"
m.screenshot(shot)
a = np.array(Image.open(shot).convert("RGB")).astype(int)
H, W, _ = a.shape
R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]

# Big near-white rectangle = the textarea. Look in the composer band.
white = (R > 235) & (G > 235) & (B > 235)
cands = []
for y in range(700, min(H, 1000), 6):
    run = white[y]
    xs = np.where(run)[0]
    if len(xs) < 200:
        continue
    # widest contiguous white run on this row
    splits = np.split(xs, np.where(np.diff(xs) > 3)[0] + 1)
    for s in splits:
        if len(s) > 300:
            cands.append((y, int(s[0]), int(s[-1]), len(s)))

print("white rows found:", len(cands))
groups = {}
for y, x0, x1, n in cands:
    key = (x0 // 40, x1 // 40)
    groups.setdefault(key, []).append((y, x0, x1, n))

targets = []
for key, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    ys = [r[0] for r in rows]
    if len(rows) < 3:
        continue
    x0 = int(np.median([r[1] for r in rows])); x1 = int(np.median([r[2] for r in rows]))
    cx, cy = (x0 + x1) // 2, (min(ys) + max(ys)) // 2
    targets.append((cx, cy, x0, x1, min(ys), max(ys), len(rows)))
    print("  box cx=%d cy=%d x[%d..%d] y[%d..%d] rows=%d" % (cx, cy, x0, x1, min(ys), max(ys), len(rows)))

targets = targets[:4] + [(760, 880, 0, 0, 0, 0, 0), (760, 900, 0, 0, 0, 0, 0)]

ps("Set-Clipboard -Value '%s'" % SENT); time.sleep(0.3)
for (cx, cy) in [(t[0], t[1]) for t in targets]:
    if "1063" not in (m.active_window().get("title") or ""):
        print("focus lost, abort"); sys.exit(4)
    m.click(int(cx), int(cy)); time.sleep(0.5)
    m.keys("ctrl", "a"); m.keys("DELETE"); time.sleep(0.2)
    ps("Set-Clipboard -Value '%s'" % SENT); time.sleep(0.2)
    m.keys("ctrl", "v"); time.sleep(0.9)
    ps("Set-Clipboard -Value '%s'" % POISON); time.sleep(0.3)
    m.keys("ctrl", "a"); m.keys("ctrl", "c"); time.sleep(0.8)
    got = get_clip()
    ok = got == SENT
    print("probe (%4d,%4d) -> %s %r" % (cx, cy, "ACCEPTS TEXT" if ok else "no", got[:30]))
    if ok:
        m.keys("ctrl", "a"); m.keys("DELETE")
        print("WINNER: %d,%d" % (cx, cy))
        break
