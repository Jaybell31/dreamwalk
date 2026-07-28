"""Navigate the HOST Chrome to the real erdos 1063 discussion thread and report
login state + composer geometry. NAVIGATION + RECON ONLY -- posts nothing.

Why this exists: post_comment.py asserted "1063" in the window title and then
stopped at READY without ever clicking Post. The commit message claimed the
comment was posted; the public thread shows only rickyc. Never trust the title
assert alone -- confirm the URL and the logged-in username.
"""
import sys, time, subprocess, json
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

URL = "https://www.erdosproblems.com/forum/discuss/1063"

def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                          capture_output=True).stdout.decode("utf-8", "replace")

m = MeatSuitClient()

# Pick a REAL Chrome window (not the off-screen one at x=-3000).
cands = []
for w in m.list_windows():
    r = w.get("rect") or [0, 0, 0, 0]
    L, T, R, B = r
    if L > -2000 and (R - L) > 600 and (B - T) > 400:
        cands.append((w.get("hwnd"), w.get("title", ""), r))

if not cands:
    print("NO USABLE WINDOW"); sys.exit(2)

hwnd, title, rect = cands[0]
print("targeting hwnd=%s %r rect=%s" % (hwnd, title[:60], rect))

m.focus_attach(hwnd); time.sleep(0.6)
act = m.active_window()
print("active now:", act.get("title", "")[:70])

# Address bar -> the discussion thread.
m.keys("ctrl", "l"); time.sleep(0.4)
m.type(URL); time.sleep(0.3)
m.keys("RETURN")
time.sleep(6.0)

print("after nav:", m.active_window().get("title", "")[:80])
m.screenshot("/tmp/erd_thread.png")
print("screenshot /tmp/erd_thread.png")
