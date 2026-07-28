"""Focus host Chrome, navigate to the 1063 thread, ASSERT, then OCR for login
state + composer. Recon only -- types a URL, never a comment, never Enter on a
form. Combined into ONE script so no other automation can steal focus between
the navigate and the assert (that gap is what produced the 'New Tab' read).
"""
import sys, time, subprocess
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

URL = "https://www.erdosproblems.com/forum/discuss/1063"
HWND = 983870
m = MeatSuitClient()

m.focus_attach(HWND); time.sleep(0.7)
m.keys("ctrl", "l"); time.sleep(0.5)
m.type(URL); time.sleep(0.4)
m.keys("RETURN")
time.sleep(8.0)

title = (m.active_window().get("title") or "")
print("TITLE:", title[:90])
if "1063" not in title:
    print("ASSERT FAILED: not on 1063. Sending nothing further.")
    sys.exit(4)
print("ASSERT OK")

m.screenshot("/tmp/erd_top.png")
m.keys("END"); time.sleep(2.5)
title2 = (m.active_window().get("title") or "")
if "1063" not in title2:
    print("focus LOST after scroll:", title2[:70]); sys.exit(5)
m.screenshot("/tmp/erd_end.png")

for shot in ("/tmp/erd_top.png", "/tmp/erd_end.png"):
    out = shot.replace(".png", "_ocr")
    subprocess.run(["tesseract", shot, out], capture_output=True)
    try:
        txt = open(out + ".txt", encoding="utf-8", errors="replace").read()
    except Exception:
        txt = ""
    flat = " ".join(txt.split())
    print("\n===== %s =====" % shot)
    print(flat[:1100])
    low = flat.lower()
    print("-- jbell31 present:", "jbell31" in low)
    print("-- login prompt   :", any(k in low for k in ("log in", "sign in", "login")))
    print("-- composer words :", [k for k in ("add a comment", "post comment", "your comment",
                                              "write", "reply", "comment") if k in low])
