#!/usr/bin/env python3
"""T27 -- GEMINI'S NONCE-ENGLISH TEST. Kills the "user error" dismissal.

THE DISMISSAL (Gemini, R12-C, named as the most damaging remaining line):
  "You are testing English-only models on out-of-distribution CJK text. This
   is USER ERROR, not a model defect. Your own bge-m3 control proves a properly
   localized model handles it. WORKING AS INTENDED."
If true, the whole finding reduces to "don't use an English model on Chinese"
and nobody cares.

HIS TEST: use pure ASCII ENGLISH strings that the tokenizer cannot represent
distinctly. If ASCII English ALSO collapses, the defect is the tokenizer's
UNHANDLED FALLBACK ARCHITECTURE, not a language mismatch -- and "use a
multilingual model" stops being an answer, because bge-m3 will not save you
from an English string either.

WHAT COUNTS AS COLLAPSE HERE. WordPiece never emits [UNK] for ASCII -- it
always shatters into subwords. So identical vectors are NOT expected. The
honest question is whether SUBWORD FRACTURE destroys distinguishability the
same way collision does. We therefore measure the SAME endpoints as the CJK
work, on three arms:
  REAL      ordinary English words (positive control -- must be near ceiling)
  NONCE     invented pronounceable words (Binglebop, Snorklewacker...)
  FRACTURED nonce words built from rare character runs, maximally shattered
and we report tokens-per-word so the fracture is visible, plus R@1 and the
ceiling-free margin from T25.

PRE-REGISTERED (before running):
  * If NONCE/FRACTURED R@1 stays near the REAL arm, the tokenizer handles
    unseen ASCII gracefully -> the CJK failure IS language-specific, Gemini's
    attack SUCCEEDS, and we must narrow the paper to a coverage claim.
  * If NONCE/FRACTURED collapse toward chance with negative margins, the
    defect is the FALLBACK PATH itself and the finding generalises beyond CJK.
  ABORT the "general fallback defect" claim if fractured-arm R@1 >= 0.70 x
  real-arm R@1 in >= 2 of 3 models.
  bge-m3 is included: if it ALSO fails on fractured ASCII, "use a multilingual
  model" is definitively not a fix.
"""
import json
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t27_nonce.json"
MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base",
          "BAAI/bge-m3"]

# 20 entities. Each has a REAL name, a NONCE name, a FRACTURED name, one doc.
DATA = [
 ("hospital",  "Binglebop",     "Xqvzhbrongk",  "Doctors and nurses treat injured patients here, operating theatres run through the night, ambulances arrive at the emergency entrance, and wards hold recovering patients."),
 ("airport",   "Snorklewack",   "Zzqrfthwlmp",  "Passengers check in luggage and pass through security screening, aircraft taxi to the runway for departure, and arrival boards list incoming international flights."),
 ("library",   "Frumbledorf",   "Vkxjzqwrthn",  "Readers borrow books from long shelves, students study quietly at wooden desks, librarians catalogue new acquisitions, and reference volumes stay in the building."),
 ("bakery",    "Wobblethunk",   "Qhzxvbnmrkl",  "Fresh bread is baked in ovens before dawn, pastries and cakes fill the display counter, and the smell of yeast drifts onto the street each morning."),
 ("shipyard",  "Grumblesnitch", "Jkxqzvwbrht",  "Steel hulls are welded on the slipway, cranes lift engine blocks into position, and completed vessels are launched into the deep water channel."),
 ("vineyard",  "Plimberwash",   "Xzqkvbnjrmw",  "Grapes ripen on trellised vines across the slope, pickers harvest fruit in the autumn, and juice ferments in oak barrels in the cellar below."),
 ("foundry",   "Krumbleflax",   "Wqzxjvbnrkm",  "Molten metal is poured into sand moulds, castings cool on the workshop floor, and finished components are ground smooth before inspection."),
 ("orchard",   "Dinglewhort",   "Zvqxwkjbnrm",  "Apple and pear trees stand in long rows, blossom appears in spring, and ladders lean against the branches when the fruit is picked in autumn."),
 ("quarry",    "Snufflebrick",  "Qxzvwjknbrm",  "Explosives loosen rock from the working face, dump trucks haul stone to the crusher, and graded aggregate is stockpiled for road building."),
 ("aviary",    "Thrumblepick",  "Vzxqwkbnjrm",  "Hundreds of birds live inside tall mesh enclosures, keepers scatter seed each morning, and visitors walk a path between the flight cages."),
 ("brewery",   "Clunkerbosh",   "Xwqzvkjbnrm",  "Malted barley is mashed in copper vessels, hops are added during the boil, and the finished beer conditions in tanks before it is kegged."),
 ("tannery",   "Splodgewick",   "Kqxzvwjbnrm",  "Raw hides are soaked in pits of solution, workers scrape away hair and flesh, and treated leather is hung on frames to dry in the airy loft."),
 ("smithy",    "Grindlepuff",   "Zqwxvkjbnrm",  "Iron is heated in the forge until it glows, the smith hammers it on the anvil, and finished horseshoes are quenched in a barrel of water."),
 ("cannery",   "Bloopersnag",   "Wxqzvkjbnrm",  "Fish are gutted and packed into tins on the line, lids are sealed under pressure, and labelled cases are stacked for shipment to wholesalers."),
 ("hatchery",  "Munglewhisk",   "Qzxwvkjbnrm",  "Eggs incubate in warm trays, chicks emerge under heat lamps, and young birds are moved to rearing sheds once they are strong enough."),
 ("granary",   "Frobblesnee",   "Vqxzwkjbnrm",  "Harvested grain is dried and stored in tall silos, augers move it between bins, and lorries collect loads for the flour mill through the winter."),
 ("dairy",     "Twizzlebonk",   "Xzqvwkjbnrm",  "Cows are milked twice daily in the parlour, milk is chilled in bulk tanks, and butter and cheese are made in the adjoining processing room."),
 ("sawmill",   "Blunderfitch",  "Zwqxvkjbnrm",  "Logs are debarked and fed through the head saw, planks are stacked to season, and sawdust is collected by extraction ducts overhead."),
 ("apiary",    "Snickerplop",   "Qwxzvkjbnrm",  "Bee colonies live in stacked wooden hives, keepers wear veils when inspecting frames, and honey is spun out in the extraction shed each summer."),
 ("pottery",   "Chunklewisp",   "Vxqzwkjbnrm",  "Clay is thrown on the wheel and shaped by hand, glazed pieces are stacked in the kiln, and fired ceramics cool slowly before they are sold."),
]


def margins(Q, D, idx):
    S = Q @ D.T
    gold = S[idx, idx].copy()
    S2 = S.copy(); S2[idx, idx] = -np.inf
    return gold - S2.max(1)


def main():
    docs = [d[3] for d in DATA]
    arms = {"real": [d[0] for d in DATA],
            "nonce": [d[1] for d in DATA],
            "fractured": [d[2] for d in DATA]}
    n = len(DATA); idx = np.arange(n)

    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    print("tokens per word (WordPiece, 30522 vocab):")
    for a, words in arms.items():
        tpw = np.mean([len(tk(w, add_special_tokens=False)["input_ids"]) for w in words])
        unk = sum(1 for w in words if 100 in tk(w, add_special_tokens=False)["input_ids"])
        print(f"  {a:10} mean tokens/word = {tpw:.2f}   words containing [UNK] = {unk}/{n}")
        ex = words[0]
        print(f"             e.g. {ex!r} -> {tk.convert_ids_to_tokens(tk(ex, add_special_tokens=False)['input_ids'])}")

    res = {"n": n, "models": {},
           "gate": "abort general-fallback claim if fractured R@1 >= 0.70 x real R@1 in >=2/3 EN models"}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        out = {}
        for a, words in arms.items():
            Q = m.encode(words, normalize_embeddings=True, show_progress_bar=False)
            hit = ((Q @ D.T).argmax(1) == idx)
            mg = margins(Q, D, idx)
            out[a] = {"r1": float(hit.mean()), "margin": float(mg.mean())}
            print(f"  {a:10} R@1={out[a]['r1']:.3f}   margin={out[a]['margin']:+.4f}")
        ratio = out["fractured"]["r1"] / out["real"]["r1"] if out["real"]["r1"] > 0 else float("nan")
        out["fractured_over_real"] = float(ratio)
        print(f"  fractured/real = {ratio:.3f}")
        res["models"][mname] = out
        del m

    en = [v for k, v in res["models"].items() if "m3" not in k]
    n_ok = sum(1 for v in en if v["fractured_over_real"] >= 0.70)
    res["verdict"] = ("ABORT general-fallback claim -- ASCII handled gracefully, CJK failure is language-specific"
                      if n_ok >= 2 else
                      "GENERAL FALLBACK DEFECT -- unseen ASCII English also collapses")
    print(f"\n{n_ok}/3 English-vocab models kept fractured >= 0.70 x real")
    print(f"VERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
