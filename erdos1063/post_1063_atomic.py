"""ATOMIC: scroll to composer, RE-MEASURE, click, paste, verify, POST, verify.

Root cause of the earlier failures: this page is ~11 comments of MathJax. It
re-renders and GROWS after END, so the composer coordinates measured in one
screenshot are stale by the time of the next call -- clicks landed in the math
body. Fix: measure the "Write your comment here" placeholder and click it in the
SAME iteration, and re-verify before each escalation.
"""
import sys, time, subprocess, csv
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

SRC_WSL = "/home/jason/dreamwalk/erdos1063/forum_comment_ascii.txt"
SRC_WIN = r"C:\Users\jason\Desktop\erdos1063_comment.txt"
POISON = "CLIP_POISON_9f3a"
HWND = 983870
DO_POST = "--post" in sys.argv


def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                          capture_output=True).stdout.decode("utf-8", "replace")


def clip():
    return ps("Get-Clipboard -Raw").replace("\r\n", "\n").strip()


def norm(s):
    return s.replace("\r\n", "\n").strip()


m = MeatSuitClient()


def guard(tag):
    t = (m.active_window().get("title") or "")
    if "1063" not in t:
        print("ABORT(%s): %r" % (tag, t[:60])); sys.exit(4)


def words(shot):
    subprocess.run(["tesseract", shot, shot[:-4], "tsv"], capture_output=True)
    out = []
    try:
        for r in csv.DictReader(open(shot[:-4] + ".tsv", encoding="utf-8",
                                     errors="replace"), delimiter="\t"):
            t = (r.get("text") or "").strip()
            if not t:
                continue
            try:
                if float(r.get("conf", -1)) < 30:
                    continue
                L, T, W, H = int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"])
            except Exception:
                continue
            out.append((t, L, T, W, H))
    except Exception:
        pass
    return out


def anchors():
    """Return (textarea_xy, postbtn_xy) from a FRESH screenshot, or (None,None)."""
    shot = "/tmp/erd_live.png"
    m.screenshot(shot)
    ws = words(shot)
    flat = " ".join(w[0] for w in ws)
    ta = btn = None
    for t, L, T, W, H in ws:
        low = t.lower().strip(".,:()")
        if low == "write" and ta is None:
            ta = (L + 180, T + 45)          # inside the box, below placeholder line 1
        # "Post comment" button: the two words sit on the same OCR row.
        if low == "post":
            for t2, L2, T2, W2, H2 in ws:
                if (t2.lower().strip(".,:()") == "comment"
                        and abs(T2 - T) < 14 and 0 < (L2 - L) < 120):
                    btn = ((L + L2 + W2) // 2, T + H // 2)
    # Fallback: the submit button sits just under the textarea.
    if ta and not btn:
        btn = (ta[0], ta[1] + 150)
    return ta, btn, flat


src = open(SRC_WSL, encoding="utf-8").read()
m.focus_attach(HWND); time.sleep(0.9)

# NAVIGATE ourselves instead of assuming the tab is still on 1063. Q uses this
# desktop; the tab was found on Google Ads / New Tab between runs. Typing a URL
# into the address bar is safe, but ONLY after we know which window we hold --
# assert the window is Chrome (has an address bar) before ctrl+L.
t0 = (m.active_window().get("title") or "")
if "Chrome" not in t0:
    print("ABORT: hwnd %s is not Chrome (%r)" % (HWND, t0[:60])); sys.exit(8)
if "1063" not in t0:
    print("tab is on %r -- navigating to the thread" % t0[:50])
    m.keys("ctrl", "l"); time.sleep(0.5)
    m.type("https://www.erdosproblems.com/forum/discuss/1063"); time.sleep(0.4)
    m.keys("RETURN"); time.sleep(9.0)
guard("start")

ta = btn = None
for attempt in range(6):
    # RE-ATTACH every iteration. Windows hands focus back to the launching
    # terminal during the 4s MathJax sleeps, which aborted attempt 3 with
    # ABORT(scroll3): 'jason@ja: ~'. focus_attach is cheap; do it every pass.
    m.focus_attach(HWND); time.sleep(0.6)
    guard("scroll%d" % attempt)
    m.keys("ctrl", "END")
    time.sleep(4.0)                 # let MathJax settle
    ta, btn, flat = anchors()
    print("attempt %d: composer_anchor=%s post_btn=%s" % (attempt, ta, btn))
    if ta and btn:
        break

if not (ta and btn):
    print("Could not locate composer after 6 tries. Nothing sent."); sys.exit(6)

guard("click")
m.click(*ta); time.sleep(0.8)
m.type("ZZ"); time.sleep(1.0)
ta2, btn2, flat2 = anchors()
if "ZZ" not in flat2.replace(" ", ""):
    print("sentinel did NOT land at %s -- nothing sent." % (ta,)); sys.exit(7)
print("FOCUS CONFIRMED in textarea at", ta)
m.keys("ctrl", "a"); m.keys("DELETE"); time.sleep(0.4)

old = clip()
try:
    ps("Set-Clipboard -Value (Get-Content -Raw '%s')" % SRC_WIN)
    if norm(clip()) != norm(src):
        print("clipboard load mismatch"); sys.exit(5)
    guard("paste")
    m.keys("ctrl", "v"); time.sleep(2.5)

    ps("Set-Clipboard -Value '%s'" % POISON); time.sleep(0.4)
    guard("readback")
    m.keys("ctrl", "a"); time.sleep(0.4); m.keys("ctrl", "c"); time.sleep(1.2)
    back = clip()
    if back == POISON:
        print("FAIL: copy no-op, composer empty. NOTHING SENT."); sys.exit(2)
    if norm(back) != norm(src):
        print("FAIL readback %d vs %d" % (len(back), len(src))); sys.exit(3)
    print("VERIFIED IN COMPOSER: %d chars exact" % len(back))

    m.click(*ta); time.sleep(0.4)
    m.screenshot("/tmp/erd_pre_submit.png")

    if not DO_POST:
        print("DRY RUN -- composer loaded, not submitted. Re-run with --post")
    else:
        _, btn3, _ = anchors()
        target = btn3 or btn
        guard("submit")
        print("CLICKING POST at", target)
        m.click(*target)
        time.sleep(9.0)
        m.screenshot("/tmp/erd_post_submit.png")
        print("submitted; shot /tmp/erd_post_submit.png")
finally:
    if old:
        ps("Set-Clipboard -Value @'\n" + old + "\n'@")
        print("clipboard restored")
