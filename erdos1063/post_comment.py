import subprocess, time, sys
sys.path.insert(0, "/home/jason/meatsuit")
from meatsuit import MeatSuitClient

SRC_WIN = r"C:\Users\jason\Desktop\erdos1063_comment.txt"
POISON  = "CLIP_POISON_9f3a"
TA      = (760, 860)     # textarea center (screen px)
BTN     = (857, 946)     # Post comment button

def ps(cmd):
    return subprocess.run(["powershell.exe","-NoProfile","-Command",cmd],
                          capture_output=True).stdout.decode("utf-8","replace")

def get_clip():
    return ps("Get-Clipboard -Raw").replace("\r\n","\n").rstrip("\n")

def norm(s):
    return s.replace("\r\n","\n").strip()

m = MeatSuitClient()
m.focus_attach(983870); time.sleep(0.4)
title = m.active_window()["title"]
assert "1063" in title, f"WRONG WINDOW: {title}"
print("window OK:", title)

# --- 1. save Q's clipboard ---
old = get_clip()
print("saved Q clipboard, %d chars" % len(old))

try:
    src = open("/home/jason/dreamwalk/erdos1063/forum_comment_ascii.txt").read()

    # --- 2. load payload ---
    ps(f"Set-Clipboard -Value (Get-Content -Raw '{SRC_WIN}')")
    got = get_clip()
    assert norm(got) == norm(src), "clipboard load mismatch %d vs %d" % (len(got), len(src))
    print("clipboard loaded + byte-verified:", len(src), "chars")

    # --- 3. click textarea, paste ---
    m.click(*TA); time.sleep(0.6)
    m.keys("ctrl","a"); m.keys("DELETE"); time.sleep(0.3)
    m.keys("ctrl","v"); time.sleep(1.5)

    # --- 4. POISON then read back ---
    ps(f"Set-Clipboard -Value '{POISON}'")
    time.sleep(0.3)
    m.keys("ctrl","a"); time.sleep(0.3)
    m.keys("ctrl","c"); time.sleep(0.8)
    back = get_clip()
    if norm(back) == POISON:
        print("FAIL: copy never happened -- composer empty or not focused. NOTHING SENT.")
        sys.exit(2)
    if norm(back) != norm(src):
        print("FAIL: readback mismatch %d vs %d" % (len(back), len(src)))
        print("HEAD:", repr(back[:120]))
        sys.exit(3)
    print("VERIFIED IN COMPOSER:", len(back), "chars -- exact match")

    # deselect so ctrl+a selection doesn't get clobbered by the click
    m.click(*TA); time.sleep(0.3)
    m.screenshot("/tmp/erd_pre_submit.png")
    print("READY. screenshot /tmp/erd_pre_submit.png")

finally:
    ps("Set-Clipboard -Value @'\n" + old + "\n'@") if old else None
    print("clipboard restored")
