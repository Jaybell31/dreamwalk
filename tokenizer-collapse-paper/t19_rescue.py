#!/usr/bin/env python3
"""T19 -- SAME-ENTITY COLLISION RESCUE. Julius's ORIGINAL requested design.

WHY THIS EXISTS. Julius asked in R9 for a within-model rescue: take the SAME
entity, replace only the colliding surface form with a meaning-equivalent form
that tokenizes distinctly, change nothing else. I ran T17 instead -- a natural
experiment across DIFFERENT words -- and set my own gate at 0.30 where his was
0.50. In R10 he caught both moves:
  (a) "the preregistered gate was dR@1 >= 0.50 ... it missed by 0.157"
  (b) "vocabulary membership was not manually assigned, but it was NOT
      RANDOMIZED. CJK tokens included in the 30,522-token vocabulary may
      systematically be more frequent or better represented during training."
(b) is the real hole. T17's English-query control equalises CONCEPT difficulty
but NOT token-level training quality: 山 may simply be a better-trained token
than 岳 for reasons beyond collision.

T19 closes it. The rescue pair is THE SAME REFERENT, and for the strongest
subset it is THE SAME WORD in two orthographies:
    车 (simplified) -> [UNK]        車 (traditional) -> [1954]
    马 -> [UNK]                     馬 -> [1980]
    门 -> [UNK]                     門 -> [1968]
    东 -> [UNK]                     東 -> [1879]
    树 -> [UNK]                     木 -> [1875]
Identical meaning, identical concept, identical document, identical model.
The ONLY difference is whether that orthographic form survives tokenization.
Frequency/training-quality confounds cannot explain a gap between two spellings
of one word, because the DOCUMENT and the CONCEPT are held fixed and each pair
is scored against itself.

PRE-REGISTERED, AND WE USE JULIUS'S NUMBER THIS TIME, NOT OUR OWN:
  PRIMARY GATE (Julius, R9): paired dR@1 = R@1(rescued) - R@1(collided) >= 0.50
     -> collision-specific causation SUPPORTED at his threshold.
  SECONDARY (ours, R9): dR@1 >= 0.30 with McNemar p < 0.05.
  BOTH ARE REPORTED WHETHER THEY PASS OR FAIL. If the 0.50 gate misses we say
  it missed, in the paper, in this file, next to the number.
  VOID unless the English-query control on the same corpus >= 0.70.

ORTHOGRAPHY SUBSET is reported separately -- it is the tightest control and the
one a reviewer should look at first.
"""
import json, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t19_rescue.json"

# (english, collided_form, rescued_form, same_word_orthography?, document)
PAIRS = [
 ("car",       "车",   "車", True,  "This wheeled vehicle is driven along roads, its engine burns fuel, passengers sit inside wearing seat belts, and it is parked at the kerb."),
 ("horse",     "马",   "馬", True,  "This four legged animal gallops across the paddock, is saddled and ridden by its owner, eats hay and oats, and is shod with iron shoes."),
 ("gate",      "门",   "門", True,  "This hinged barrier stands in the boundary wall, swings open to let people through, is bolted shut at night, and creaks upon rusted hinges."),
 ("east",      "东",   "東", True,  "This is the compass direction where the sun rises each morning, opposite to west, and lies to the right when facing north on a map."),
 ("study",     "學",   "学", True,  "Pupils learn and revise their subjects here, read textbooks and take notes, prepare for examinations, and attend lessons given by teachers."),
 ("wind",      "风",   "風", True,  "Moving air gusts across the landscape, bends the branches of trees, fills sails and turns turbines, and scatters leaves along the street."),
 ("university","大學", "大学", True, "Undergraduates attend lectures here and sit final examinations, researchers publish academic papers, and degrees are awarded at graduation."),
 # REMOVED: ("bank","银行","銀行") -- the design validator rejected it. BOTH the
 # simplified and traditional forms tokenize to [100, 1945], i.e. both collide.
 # It is not a rescue pair at all. Kept as a comment because it is a useful
 # negative: orthographic variation does NOT reliably rescue, which is itself
 # evidence that in-vocab CJK coverage is arbitrary rather than systematic.

 ("tree",      "树",   "木", False, "Branches and green leaves grow from a tall wooden trunk, roots spread underground, and birds nest among the boughs in the spring."),
 ("mountain",  "岳",   "山", False, "Climbers ascend the steep rocky slopes, hikers follow trails to the summit, and snow covers the high peaks and ridges through the winter."),
 ("sea",       "洋",   "海", False, "Salt water stretches to the horizon, waves break upon the shore, ships cross the open expanse, and gulls circle above the swell."),
 ("fire",      "炎",   "火", False, "Flames burn brightly and give off heat and smoke, wood crackles as it is consumed, and embers glow red long after the blaze dies down."),
 ("stone",     "岩",   "石", False, "This hard grey mineral lump is quarried and cut into blocks, used to build walls and roads, and skips across water when thrown flat."),
 ("dog",       "狗",   "犬", False, "This loyal four legged pet barks at strangers, wags its tail when greeted, is taken out on a lead for walks, and fetches a thrown ball."),
 ("book",      "书",   "本", False, "Printed pages are bound between covers, readers turn them one by one, chapters follow in order, and it sits on a shelf when finished."),
 ("gold",      "黄金", "金", False, "This heavy yellow metal is mined from the ground, refined into bars and coins, worked into rings and necklaces, and stored in secure vaults."),
 ("moon",      "月亮", "月", False, "This pale body orbits our planet and shines at night, passing through crescent and full phases, and its pull raises and lowers the ocean tides."),
 ("sun",       "太阳", "日", False, "This bright star rises in the east each morning, warms the ground through the day, casts long shadows at evening, and sets below the horizon."),
 ("school",    "学校", "学", False, "Children attend lessons in classrooms here, teachers set homework and mark examinations, and pupils progress through year groups to graduation."),
 ("hotel",     "酒店", "宿", False, "Guests check into rooms here for the night, staff clean and prepare the bedrooms daily, and travellers leave their bags at reception on arrival."),
 ("restaurant","餐馆", "食堂", False,"Diners order meals from a menu here, chefs prepare dishes in the kitchen, and waiters bring food and drink to tables throughout the evening."),
 ("farm",      "农场", "田", False, "Crops are planted and harvested in the surrounding fields, and livestock such as cattle and chickens are raised for milk, eggs and meat."),
 ("hall",      "館",   "堂", False, "People gather inside this large public building for meetings and ceremonies, rows of seats face a platform, and its doors open onto a foyer."),
]

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base",
          "BAAI/bge-m3"]

def mcnemar(b, c):
    """Exact two-sided McNemar. b = rescued-only wins, c = collided-only wins."""
    from math import comb
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))

def main():
    docs = [p[4] for p in PAIRS]
    n = len(PAIRS)
    idx = np.arange(n)
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # VALIDATE the design before scoring anything: collided must contain [UNK],
    # rescued must contain none. If this fails the whole test is meaningless.
    bad = []
    for en, coll, resc, _, _ in PAIRS:
        ci = tk(coll, add_special_tokens=False)["input_ids"]
        ri = tk(resc, add_special_tokens=False)["input_ids"]
        if 100 not in ci or 100 in ri:
            bad.append((en, coll, ci, resc, ri))
    print(f"DESIGN VALIDATION: {len(PAIRS)-len(bad)}/{len(PAIRS)} pairs correctly formed")
    if bad:
        print("  MALFORMED (test is VOID for these):", bad)
        return
    res = {"n_pairs": n, "malformed": len(bad), "models": {},
           "gates": {"julius_primary": 0.50, "ours_secondary": 0.30}}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        arms = {}
        for key, col in (("collided", 1), ("rescued", 2), ("english", 0)):
            Q = m.encode([p[col] for p in PAIRS], normalize_embeddings=True, show_progress_bar=False)
            arms[key] = ((Q @ D.T).argmax(1) == idx)
        c_hits, r_hits, e_hits = arms["collided"], arms["rescued"], arms["english"]
        delta = float(r_hits.mean() - c_hits.mean())
        b = int(np.sum(r_hits & ~c_hits)); c = int(np.sum(c_hits & ~r_hits))
        p = mcnemar(b, c)

        ortho = [i for i, pr in enumerate(PAIRS) if pr[3]]
        od = float(np.mean(r_hits[ortho]) - np.mean(c_hits[ortho]))

        valid = float(e_hits.mean()) >= 0.70
        out = {"collided_r1": float(c_hits.mean()), "rescued_r1": float(r_hits.mean()),
               "delta": delta, "mcnemar_b_rescued_only": b, "mcnemar_c_collided_only": c,
               "mcnemar_p": float(p), "english_control": float(e_hits.mean()),
               "valid": valid,
               "orthography_subset": {"n": len(ortho), "delta": od,
                                      "collided": float(np.mean(c_hits[ortho])),
                                      "rescued": float(np.mean(r_hits[ortho]))},
               "julius_gate_0.50": "PASS" if (valid and delta >= 0.50) else "MISS",
               "our_gate_0.30": "PASS" if (valid and delta >= 0.30 and p < 0.05) else "MISS"}
        print(f"  collided {out['collided_r1']:.3f} -> rescued {out['rescued_r1']:.3f}"
              f"   delta {delta:+.3f}  McNemar p={p:.5f} (b={b}, c={c})")
        print(f"  english control {out['english_control']:.3f}  ({'VALID' if valid else 'VOID'})")
        print(f"  same-word orthography subset (n={len(ortho)}): "
              f"{out['orthography_subset']['collided']:.3f} -> {out['orthography_subset']['rescued']:.3f}  delta {od:+.3f}")
        print(f"  JULIUS GATE (>=0.50): {out['julius_gate_0.50']}   OUR GATE (>=0.30, p<.05): {out['our_gate_0.30']}")
        res["models"][mname] = out
        del m

    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
