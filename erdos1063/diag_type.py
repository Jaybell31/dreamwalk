"""One-step diagnostic: does a click+type actually land in the 1063 textarea?

The clipboard probe reported POISON at every candidate (= ctrl+c copied nothing),
yet keys() demonstrably work (END scrolled the page). So test the simplest thing:
click, type a sentinel with m.type (no clipboard at all), screenshot, OCR.
"""
import sys, time, subprocess, csv
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

HWND = 983870
SENT = "ZZPROBEZZ"
m = MeatSuitClient()
m.focus_attach(HWND); time.sleep(0.8)
if "1063" not in (m.active_window().get("title") or ""):
    print("ABORT: not on 1063"); sys.exit(4)

m.keys("END"); time.sleep(2.0)

for (cx, cy) in [(700, 780), (600, 800), (860, 800), (500, 760)]:
    m.click(cx, cy); time.sleep(0.6)
    m.type(SENT); time.sleep(1.2)
    shot = "/tmp/erd_diag_%d_%d.png" % (cx, cy)
    m.screenshot(shot)
    subprocess.run(["tesseract", shot, shot[:-4], "tsv"], capture_output=True)
    txt = ""
    try:
        for r in csv.DictReader(open(shot[:-4] + ".tsv", encoding="utf-8",
                                     errors="replace"), delimiter="\t"):
            t = (r.get("text") or "").strip()
            if t:
                txt += t + " "
    except Exception:
        pass
    landed = "ZZPROBE" in txt.replace(" ", "").upper()
    print("click (%d,%d) type -> landed=%s" % (cx, cy, landed))
    if landed:
        m.keys("ctrl", "a"); m.keys("DELETE"); time.sleep(0.3)
        print("WINNER TEXTAREA: (%d,%d)" % (cx, cy))
        break
