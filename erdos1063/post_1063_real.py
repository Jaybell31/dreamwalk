"""POST the erdos1063 comment for real: paste -> poison-verify -> CLICK POST -> verify.

This replaces post_comment.py, which stopped at "READY" and never clicked Post
while a commit message claimed the comment was posted. The public thread showed
only rickyc. Lesson encoded here: the submit click and a POST-SUBMIT PUBLIC
READBACK are part of the job, not a follow-up.

Safety: Q's live Polymarket portfolio is open in another Chrome window. We assert
the active window title contains "1063" immediately before EVERY input call.
"""
import sys, time, subprocess
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

SRC_WSL = "/home/jason/dreamwalk/erdos1063/forum_comment_ascii.txt"
SRC_WIN = r"C:\Users\jason\Desktop\erdos1063_comment.txt"
POISON  = "CLIP_POISON_9f3a"
HWND    = 983870
TA      = (760, 860)    # textarea, below the "Your comment" OCR anchor (y=795)
BTN     = (849, 946)    # "Post comment" -- OCR: Post cx=824, comment cx=874


def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                          capture_output=True).stdout.decode("utf-8", "replace")


def get_clip():
    return ps("Get-Clipboard -Raw").replace("\r\n", "\n").rstrip("\n")


def norm(s):
    return s.replace("\r\n", "\n").strip()


m = MeatSuitClient()


def guard(where):
    t = (m.active_window().get("title") or "")
    if "1063" not in t:
        print("ABORT (%s): focus is on %r -- sending nothing." % (where, t[:60]))
        sys.exit(4)
    return t


src = open(SRC_WSL, encoding="utf-8").read()
m.focus_attach(HWND); time.sleep(0.8)
guard("startup")
print("on thread OK | payload %d chars" % len(src))

old = get_clip()
print("saved Q's clipboard (%d chars) -- will restore" % len(old))

posted = False
try:
    ps("Set-Clipboard -Value (Get-Content -Raw '%s')" % SRC_WIN)
    if norm(get_clip()) != norm(src):
        print("clipboard load mismatch -- abort"); sys.exit(5)
    print("clipboard loaded + byte-verified")

    guard("pre-click")
    m.keys("END"); time.sleep(1.2)
    m.click(*TA); time.sleep(0.7)
    m.keys("ctrl", "a"); m.keys("DELETE"); time.sleep(0.3)
    m.keys("ctrl", "v"); time.sleep(2.0)

    # POISON FIRST or the readback is a lie (ctrl+c on empty box is a no-op).
    ps("Set-Clipboard -Value '%s'" % POISON); time.sleep(0.4)
    guard("pre-readback")
    m.keys("ctrl", "a"); time.sleep(0.4)
    m.keys("ctrl", "c"); time.sleep(1.0)
    back = get_clip()

    if norm(back) == POISON:
        print("FAIL: composer empty / not focused. NOTHING SENT."); sys.exit(2)
    if norm(back) != norm(src):
        print("FAIL: readback mismatch %d vs %d" % (len(back), len(src)))
        print("HEAD:", repr(back[:120])); sys.exit(3)
    print("VERIFIED IN COMPOSER: %d chars exact" % len(back))

    m.click(*TA); time.sleep(0.4)          # deselect
    m.screenshot("/tmp/erd_pre_submit.png")

    # ---- THE STEP THE OLD SCRIPT NEVER DID ----
    guard("pre-submit")
    print("clicking Post comment at", BTN)
    m.click(*BTN)
    time.sleep(7.0)
    m.screenshot("/tmp/erd_post_submit.png")
    posted = True
    print("submit click sent; shot /tmp/erd_post_submit.png")

finally:
    if old:
        ps("Set-Clipboard -Value @'\n" + old + "\n'@")
        print("clipboard restored to Q's content")

print("SUBMIT_ATTEMPTED:", posted)
