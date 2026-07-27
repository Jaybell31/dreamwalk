#!/usr/bin/env python3
"""T17 -- WITHIN-MODEL COLLISION CONTRAST. Answers Julius's kill shot (R9).

THE OBJECTION WE ARE ANSWERING (Julius, devil's seat, R9 -- and he is right):
  T14 compared collapsed models (0.083) against bge-m3 (0.958) and attributed
  the gap to [UNK] collision. But bge-m3 differs in vocabulary AND multilingual
  training AND representation quality all at once. "Because bge-m3
  simultaneously changes tokenizer behavior, multilingual training, vocabulary,
  and representation quality, the comparison cannot attribute the gap
  specifically to collisions." A cross-MODEL gap is SEPARATION, not CAUSATION.
  The rival explanation -- "English-oriented models are simply bad at Chinese,
  collision or not" -- is not excluded by T14.

THE FIX: never leave the model. Contrast WITHIN one model, WITHIN one language.
The 30522 WordPiece vocab contains 488 CJK-bearing tokens. So some Chinese
words tokenize to REAL DISTINCT ids and others collapse to [UNK]. That gives us
a natural experiment the cross-model design could never give:

    DISTINCT arm   山 mountain -> [1831]        real, unique token
    COLLIDED arm   雨 rain     -> [100]         [UNK]

Both arms are Chinese. Both go through the SAME model, SAME corpus, SAME task,
SAME scoring. Cross-lingual difficulty, tokenizer, training data and
representation quality are held CONSTANT BY CONSTRUCTION. The only thing that
varies is whether the query survives tokenization.

  If "English models are just bad at Chinese" -> BOTH arms fail. Collision is
     incidental and the paper's causal claim dies; we narrow to representation.
  If DISTINCT retrieves and COLLIDED does not -> collision is the CAUSE.

THE CONCEPT-DIFFICULTY CONTROL (this is the load-bearing one).
A skeptic's next move: "your collided words are just harder concepts, or rarer,
or longer." So every entity is ALSO queried in ENGLISH against the same corpus.
If the two arms score the SAME in English (they should -- these are all common
concrete nouns) then the documents and concepts are equally retrievable, and
the ONLY surviving difference between the arms is Chinese query encoding.
That is the control that converts a correlation into an attribution.

PRE-REGISTERED, WRITTEN BEFORE THE RUN (edit history is in git/file mtime):
  PRIMARY GATE  delta = R@1(DISTINCT_zh) - R@1(COLLIDED_zh), per collapsed model.
     delta >= 0.30 AND Fisher exact p < 0.05  -> COLLISION-SPECIFIC CAUSATION
                                                  SUPPORTED. Causal claim stands.
     delta <  0.30                            -> GENERIC CROSS-LINGUAL
                                                  INCAPACITY DOMINATES. We strike
                                                  the causal retrieval claim and
                                                  narrow the paper to the
                                                  representational collision.
  VALIDITY GATE (test is VOID unless both hold):
     (a) English-query R@1 >= 0.70 in BOTH arms  -> corpus retrievable, concepts fair
     (b) |R@1(EN,DISTINCT) - R@1(EN,COLLIDED)| <= 0.20 -> arms equally easy in English
  ARM-BALANCE NOTE: reported both for the full sets and for a CHAR-LENGTH-MATCHED
     subset, because collided words skew to 2-char compounds. If the matched
     subset contradicts the full set, the matched subset wins.
"""
import json, itertools, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t17_within_model.json"

# (chinese, english, english_document). Arm membership is DERIVED from the
# tokenizer at runtime, never hand-assigned -- so we cannot rig the split.
ITEMS = [
 # --- expected DISTINCT (in-vocab CJK chars) ---
 ("山",  "mountain", "Climbers ascend the steep rocky slopes here, hikers follow trails to the summit, and snow covers the high peaks and ridges through the winter months."),
 ("川",  "river",    "Water flows downstream between the banks here, anglers fish from the shore, and small boats drift with the current toward the wider estuary."),
 ("水",  "water",    "This clear liquid is drunk daily, used for washing and cooking, carried through pipes into homes, and boiled in kettles for tea and coffee."),
 ("火",  "fire",     "Flames burn brightly and give off heat and smoke here, wood crackles as it is consumed, and embers glow red long after the blaze dies down."),
 ("日",  "sun",      "This bright star rises in the east each morning, warms the ground through the day, casts long shadows at evening, and sets below the horizon."),
 ("月",  "moon",     "This pale body orbits our planet and shines at night, passing through crescent and full phases, and its pull raises and lowers the ocean tides."),
 ("木",  "tree",     "Branches and green leaves grow from a tall wooden trunk here, roots spread underground, and birds nest among the boughs in the spring."),
 ("金",  "gold",     "This heavy yellow metal is mined from the ground, refined into bars and coins, worked into rings and necklaces, and stored in secure vaults."),
 ("土",  "earth",    "This brown soil is dug and turned over with a spade, seeds are sown into it, worms move through it, and rain soaks into the loose ground."),
 ("人",  "person",   "This individual human being walks and talks, holds opinions and memories, works during the day, and lives among family, friends and neighbours."),
 ("王",  "king",     "This male monarch wears a crown and rules over a kingdom, sits on a throne in the palace, and his eldest heir succeeds him upon his death."),
 ("天",  "sky",      "Clouds drift overhead across this vast blue expanse, aircraft leave white trails through it, and stars appear across it after darkness falls."),
 ("田",  "field",    "Crops are planted in rows across this open ground, tractors plough the soil in spring, and the harvest is gathered in late summer."),
 ("石",  "stone",    "This hard grey mineral lump is quarried and cut into blocks, used to build walls and roads, and skips across water when thrown flat."),
 ("花",  "flower",   "Colourful petals open on a green stem here, bees visit for nectar and spread pollen, and the blooms are cut and arranged in vases."),
 ("海",  "sea",      "Salt water stretches to the horizon here, waves break upon the shore, ships cross the open expanse, and gulls circle above the swell."),
 ("風",  "wind",     "Moving air gusts across the landscape here, bends the branches of trees, fills sails and turns turbines, and scatters leaves along the street."),
 ("馬",  "horse",    "This four legged animal gallops across the paddock, is saddled and ridden by its owner, eats hay and oats, and is shod with iron shoes."),
 ("犬",  "dog",      "This loyal four legged pet barks at strangers, wags its tail when greeted, is taken out on a lead for walks, and fetches a thrown ball."),
 ("車",  "car",      "This wheeled vehicle is driven along roads here, its engine burns fuel, passengers sit inside wearing seat belts, and it is parked at the kerb."),
 ("門",  "gate",     "This hinged barrier stands in the boundary wall, swings open to let people through, is bolted shut at night, and creaks upon rusted hinges."),
 ("家",  "house",    "People live inside this building, sleeping in its bedrooms, cooking in its kitchen, and sitting in its living room behind a front door."),
 ("国",  "country",  "This sovereign nation has borders and a capital, its citizens hold passports, its government passes laws, and it has a flag and an anthem."),
 ("学",  "study",    "Pupils learn and revise their subjects here, read textbooks and take notes, prepare for examinations, and attend lessons given by teachers."),
 ("本",  "book",     "Printed pages are bound between covers here, readers turn them one by one, chapters follow in order, and it sits on a shelf when finished."),
 ("白",  "white",    "This is the palest colour, the shade of fresh snow and clean paper and milk, reflecting all light, and the opposite of the darkest shade."),
 ("青",  "blue",     "This is the colour of a clear midday sky and of deep ocean water, sitting between green and violet in the spectrum of visible light."),
 ("東京","tokyo",    "This vast Japanese capital city has crowded rail stations, neon lit districts, a huge metropolitan population, and hosts the national government."),
 ("北京","beijing",  "This Chinese capital city contains an imperial palace complex, a vast public square, ancient walls and gates, and the seat of national government."),
 ("上海","shanghai", "This large Chinese port city on the eastern coast has a famous riverside waterfront, financial towers, and is a centre of trade and shipping."),
 ("中国","china",    "This large east Asian nation has the world's biggest population, a long imperial history, a written character script, and vast manufacturing output."),
 ("日本","japan",    "This east Asian island nation has volcanic mountains, high speed trains, an emperor as ceremonial head of state, and a long coastline."),
 ("大学","university","Undergraduates attend lectures here and sit final examinations, researchers publish academic papers, and degrees are awarded at graduation."),
 ("大",  "big",      "This describes something of great size, larger than usual, taking up much space, the opposite of little, and hard to carry in one hand."),
 ("小",  "small",    "This describes something of little size, smaller than usual, taking up hardly any space, easily held in the palm of one hand."),

 # --- expected COLLIDED (out-of-vocab CJK chars) ---
 ("医院","hospital", "Patients are admitted here for surgery and emergency treatment, and doctors and nurses provide round the clock medical care to the seriously ill."),
 ("机场","airport",  "Passengers check in their luggage here before boarding international flights, and aircraft take off and land on the runways throughout the day."),
 ("市场","market",   "Traders sell fresh produce and household goods from stalls here, and shoppers haggle over prices for vegetables, fish and spices every morning."),
 ("农场","farm",     "Crops are planted and harvested in the surrounding fields here, and livestock such as cattle and chickens are raised for milk, eggs and meat."),
 ("工厂","factory",  "Production lines assemble manufactured goods here in shifts, and machinery stamps, welds and packages components for shipment to distributors."),
 ("银行","bank",     "Customers deposit and withdraw money here, apply for mortgages and business loans, and meet advisers about savings accounts and interest rates."),
 ("学校","school",   "Children attend lessons here in classrooms, teachers set homework and mark examinations, and pupils progress through year groups to graduation."),
 ("法院","court",    "Judges hear evidence here and juries deliver verdicts, lawyers argue cases on behalf of clients, and sentences are handed down after trial."),
 ("餐馆","restaurant","Diners order meals from a menu here, chefs prepare dishes in the kitchen, and waiters bring food and drink to tables throughout the evening."),
 ("酒店","hotel",    "Guests check into rooms here for the night, staff clean and prepare the bedrooms daily, and travellers leave their bags at reception on arrival."),
 ("邮局","post office","Customers post parcels and letters here, buy stamps at the counter, and collect packages that could not be delivered to their home address."),
 ("药房","pharmacy", "Prescriptions are dispensed here by trained staff, customers buy painkillers and cold remedies, and advice is given on dosage and side effects."),
 ("超市","supermarket","Shoppers push trolleys along aisles here selecting groceries, and cashiers scan items at the checkout before customers pack their bags."),
 ("车站","station",  "Commuters wait on platforms here for trains to arrive, announcements list departure times, and tickets are checked at the barriers."),
 ("公园","park",     "Families walk along paths here among trees and flower beds, children use the playground equipment, and joggers exercise on the grass."),
 ("教堂","church",   "Congregations gather here for services and hymns, weddings and funerals are held in the main hall, and the building has stained glass windows."),
 ("剧院","theatre",  "Audiences watch live performances here from tiered seating, actors rehearse on stage before opening night, and tickets sell out for popular plays."),
 ("监狱","prison",   "Inmates are held here in cells under supervision, guards patrol the corridors and wings, and visitors are searched before entering the facility."),
 ("码头","harbour",  "Cargo ships dock here to unload containers, cranes lift freight onto the quayside, and fishing boats moor overnight alongside the pier."),
 ("雨",  "rain",     "Drops of water fall from grey clouds here, puddles form on the pavement, people open umbrellas, and the downpour drums on rooftops."),
 ("雪",  "snow",     "White flakes fall and settle in a cold layer here, footprints press into the drifts, children build figures from it, and it melts as it warms."),
 ("魚",  "fish",     "This scaled creature swims underwater using fins and a tail, breathes through gills, is caught on hooks and in nets, and is cooked and eaten."),
 ("鳥",  "bird",     "This feathered creature flies using wings, sings at dawn, builds a nest from twigs, lays eggs in it, and pecks at seeds on the ground."),
 ("猫",  "cat",      "This small furry pet purrs when stroked, hunts mice at night, washes itself with its tongue, sleeps curled up, and climbs onto high shelves."),
 ("黒",  "black",    "This is the darkest colour, the shade of coal and of a moonless night, absorbing all light, and the opposite of the palest shade."),
 ("赤",  "red",      "This is the colour of fresh blood and of ripe tomatoes, sitting at the long wavelength end of the spectrum, used on signs meaning stop."),
]

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base",
          "BAAI/bge-m3"]

def fisher(a, b, c, d):
    """Two-sided Fisher exact. a,b = arm1 hit/miss; c,d = arm2 hit/miss."""
    from math import comb
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    def p(x):
        lo = max(0, c1 - r2)
        if x < lo or x > min(r1, c1): return 0.0
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(0, min(r1, c1) + 1) if p(x) <= obs + 1e-12))

def main():
    docs = [d for _, _, d in ITEMS]
    res = {"n_items": len(ITEMS), "models": {}, "preregistered": {
        "primary_gate": "delta = R@1(DISTINCT_zh) - R@1(COLLIDED_zh) >= 0.30 and Fisher p < 0.05",
        "validity_a": "English R@1 >= 0.70 in both arms",
        "validity_b": "|R@1(EN,DISTINCT) - R@1(EN,COLLIDED)| <= 0.20"}}

    # arm assignment from the ENGLISH-vocab tokenizer, derived not hand-set
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    arms, toks = [], {}
    for zh, en, _ in ITEMS:
        ids = tk(zh, add_special_tokens=False)["input_ids"]
        toks[zh] = ids
        arms.append("COLLIDED" if 100 in ids else "DISTINCT")
    res["arms"] = {zh: {"arm": a, "ids": toks[zh], "en": en, "nchar": len(zh)}
                   for (zh, en, _), a in zip(ITEMS, arms)}
    nD = arms.count("DISTINCT"); nC = arms.count("COLLIDED")
    print(f"ARM SPLIT (derived from tokenizer): DISTINCT={nD}  COLLIDED={nC}")

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        out = {}
        for qlang in ("zh", "en"):
            qs = [zh if qlang == "zh" else en for zh, en, _ in ITEMS]
            Q = m.encode(qs, normalize_embeddings=True, show_progress_bar=False)
            hits = (Q @ D.T).argmax(1) == np.arange(len(ITEMS))
            out[qlang] = {"hits": hits.tolist(),
                          "overall": float(hits.mean()),
                          "DISTINCT": float(np.mean([h for h, a in zip(hits, arms) if a == "DISTINCT"])),
                          "COLLIDED": float(np.mean([h for h, a in zip(hits, arms) if a == "COLLIDED"]))}
        # identical-query-vector census (the mechanism, measured not asserted)
        Qzh = m.encode([zh for zh, _, _ in ITEMS], normalize_embeddings=True, show_progress_bar=False)
        coll_idx = [i for i, a in enumerate(arms) if a == "COLLIDED"]
        dist_idx = [i for i, a in enumerate(arms) if a == "DISTINCT"]
        def ident(idx):
            n = t = 0
            for i, j in itertools.combinations(idx, 2):
                t += 1; n += int(np.allclose(Qzh[i], Qzh[j], atol=1e-6))
            return f"{n}/{t}"
        out["identical_query_pairs"] = {"COLLIDED": ident(coll_idx), "DISTINCT": ident(dist_idx)}

        zh = out["zh"]
        a = sum(1 for h, ar in zip(zh["hits"], arms) if ar == "DISTINCT" and h)
        b = nD - a
        c = sum(1 for h, ar in zip(zh["hits"], arms) if ar == "COLLIDED" and h)
        d = nC - c
        delta = zh["DISTINCT"] - zh["COLLIDED"]
        p = fisher(a, b, c, d)
        out["delta_zh"] = float(delta); out["fisher_p"] = float(p)
        out["contingency"] = {"distinct_hit": a, "distinct_miss": b,
                              "collided_hit": c, "collided_miss": d}

        # length-matched subset: 2-char words only (both arms have them)
        mi = [i for i, (zz, _, _) in enumerate(ITEMS) if len(zz) == 2]
        mD = [i for i in mi if arms[i] == "DISTINCT"]; mC = [i for i in mi if arms[i] == "COLLIDED"]
        if mD and mC:
            out["matched_2char"] = {
                "n_distinct": len(mD), "n_collided": len(mC),
                "DISTINCT": float(np.mean([zh["hits"][i] for i in mD])),
                "COLLIDED": float(np.mean([zh["hits"][i] for i in mC]))}

        vA = min(out["en"]["DISTINCT"], out["en"]["COLLIDED"]) >= 0.70
        vB = abs(out["en"]["DISTINCT"] - out["en"]["COLLIDED"]) <= 0.20
        out["validity_pass"] = bool(vA and vB)
        out["verdict"] = ("VOID -- validity gate failed" if not (vA and vB) else
                          "COLLISION-SPECIFIC CAUSATION SUPPORTED" if (delta >= 0.30 and p < 0.05)
                          else "GENERIC CROSS-LINGUAL INCAPACITY DOMINATES -- narrow the claim")

        print(f"  zh: DISTINCT {zh['DISTINCT']:.3f}  COLLIDED {zh['COLLIDED']:.3f}  delta {delta:+.3f}  p={p:.5f}")
        print(f"  en: DISTINCT {out['en']['DISTINCT']:.3f}  COLLIDED {out['en']['COLLIDED']:.3f}   (validity {'PASS' if vA and vB else 'FAIL'})")
        print(f"  identical zh query pairs: COLLIDED {out['identical_query_pairs']['COLLIDED']}  DISTINCT {out['identical_query_pairs']['DISTINCT']}")
        if "matched_2char" in out:
            mm = out["matched_2char"]
            print(f"  2-char matched: DISTINCT {mm['DISTINCT']:.3f} (n={mm['n_distinct']})  COLLIDED {mm['COLLIDED']:.3f} (n={mm['n_collided']})")
        print(f"  VERDICT: {out['verdict']}")
        res["models"][mname] = out
        del m

    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
