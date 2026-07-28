"""Re-focus the 1063 thread by UNIQUE title, assert, screenshot. NO typing.

HARD LAW (meat-suit skill): Q's LIVE POLYMARKET PORTFOLIO is open in a Chrome
window on this desktop. A stray Enter/click there is real money. So:
  - match on a UNIQUE page-title substring ("1063 Discussion"), never "Chrome"
  - ASSERT active_window() contains it BEFORE any key/click
  - if the assert fails: print and EXIT NONZERO. Do not "try once more".
"""
import sys, time
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

NEED = "1063 Discussion"
m = MeatSuitClient()

hit = None
for w in m.list_windows():
    if NEED.lower() in (w.get("title") or "").lower():
        r = w.get("rect") or [0, 0, 0, 0]
        if r[0] > -2000:
            hit = w
            break

if not hit:
    print("NO WINDOW whose title contains %r." % NEED)
    print("The thread is a TAB inside a window whose active tab is something else.")
    print("Visible windows:")
    for w in m.list_windows():
        t = (w.get("title") or "")[:70]
        if t:
            print("   hwnd=%-9s %r" % (w.get("hwnd"), t))
    sys.exit(3)

m.focus_attach(hit["hwnd"]); time.sleep(0.8)
act = (m.active_window().get("title") or "")
print("active:", act[:80])
if NEED.lower() not in act.lower():
    print("ASSERT FAILED -- refusing to send any input.")
    sys.exit(4)

print("ASSERT OK -- on the 1063 thread window.")
m.screenshot("/tmp/erd_ok.png")
print("shot /tmp/erd_ok.png")
