"""Why is REAL-text raw accuracy 0.360 and not chance (0.125)?

Q's law: never dismiss an anomaly. Hypothesis: the distractor alphabet contains
❤ (U+2764), which is one of the NINE legacy typographic marks that SURVIVE the
30522 WordPiece vocab. Trials whose true emoji is a survivor stay retrievable,
lifting accuracy above chance. If so, the excess is fully explained and the
"99.83% collapse" figure is confirmed rather than undermined.
"""
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
ALPHABET = ["\U0001F602", "\u2764", "\U0001F62D", "\U0001F60D",
            "\U0001F914", "\U0001F644", "\U0001F621", "\U0001F525"]

print(f"{'emoji':8} {'ids':22} survives?")
surv = []
for e in ALPHABET:
    ids = tk(e, add_special_tokens=False)["input_ids"]
    is_unk = (len(ids) == 0) or all(i == tk.unk_token_id for i in ids)
    if not is_unk:
        surv.append(e)
    print(f"  {e:6} {str(ids):22} {'SURVIVES' if not is_unk else 'collapses'}")

n = len(ALPHABET)
k = len(surv)
print(f"\nsurvivors in alphabet: {k}/{n}  -> {surv}")
# survivor trials retrieve correctly; collapsed trials tie and argmax picks
# index 0, which is correct only when gold happens to be index 0.
pred = k / n + (1 - k / n) * (1 / n)
print(f"predicted raw accuracy if this explains it: {pred:.3f}")
print("observed: 0.360")
