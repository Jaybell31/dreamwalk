#!/usr/bin/env python3
"""T14 -- CJK RETRIEVAL, submission-grade. The Chinese claim needs a real task.

WHY: T13 proved COLLISION (医院 hospital == 机场 airport == 市场 market ==
农场 farm == 工厂 factory, all -> [100,100], one shared vector). A reviewer
will correctly say a collision count is not a retrieval failure. This runs the
same standard the emoji claim already meets: a real retrieval task with a
document gold, a positive control, and a pre-registered metric.

DESIGN. Query = a Chinese place/entity word. Corpus = ENGLISH prose documents,
one per entity, describing what happens at that place. Task: retrieve the doc
matching the query. This is realistic cross-lingual RAG: a Chinese query
against an English knowledge base, which is exactly what a multilingual
support/search deployment does.

  n = 24 entities (8 colliding + 16 additional common words)
  chance = 1/24 = 0.042
  metric = top-1 accuracy

CONTROLS (the test is void without these):
  * ENGLISH-QUERY control: same corpus, query is the English word. Every model
    must score high, proving the CORPUS is retrievable and the task is fair.
  * bge-m3 immune control: multilingual vocab must handle the Chinese queries.
  * collision-subset breakout: accuracy restricted to the 5 words that share
    ONE vector is mathematically capped at 1/5 of that group -- report it.

MITIGATION QUESTION (honest): demojize does NOT fix Chinese. We test the only
zero-retraining mitigation available to a practitioner: TRANSLITERATE/translate
the query to English before embedding. If that recovers accuracy, the paper's
advice generalises from "demojize your emoji" to "never send out-of-vocabulary
script directly to an English-vocabulary encoder".
"""
import json, numpy as np, torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t14_cjk.json"

# (chinese, english, english_document)
ITEMS = [
 ("医院", "hospital",  "Patients are admitted here for surgery and emergency treatment, and doctors and nurses provide round the clock medical care to the seriously ill."),
 ("机场", "airport",   "Passengers check in their luggage here before boarding international flights, and aircraft take off and land on the runways throughout the day."),
 ("市场", "market",    "Traders sell fresh produce and household goods from stalls here, and shoppers haggle over prices for vegetables, fish and spices every morning."),
 ("农场", "farm",      "Crops are planted and harvested in the surrounding fields here, and livestock such as cattle and chickens are raised for milk, eggs and meat."),
 ("工厂", "factory",   "Production lines assemble manufactured goods here in shifts, and machinery stamps, welds and packages components for shipment to distributors."),
 ("银行", "bank",      "Customers deposit and withdraw money here, apply for mortgages and business loans, and meet advisers about savings accounts and interest rates."),
 ("学校", "school",    "Children attend lessons here in classrooms, teachers set homework and mark examinations, and pupils progress through year groups to graduation."),
 ("法院", "court",     "Judges hear evidence here and juries deliver verdicts, lawyers argue cases on behalf of clients, and sentences are handed down after trial."),
 ("图书馆", "library",  "Visitors borrow books here with a membership card, students study quietly at desks, and archives of newspapers and journals are kept on shelves."),
 ("餐馆", "restaurant","Diners order meals from a menu here, chefs prepare dishes in the kitchen, and waiters bring food and drink to tables throughout the evening."),
 ("酒店", "hotel",     "Guests check into rooms here for the night, staff clean and prepare the bedrooms daily, and travellers leave their bags at reception on arrival."),
 ("警察局", "police station", "Officers take statements from witnesses here, suspects are questioned in interview rooms, and members of the public report crimes and thefts."),
 ("邮局", "post office","Customers post parcels and letters here, buy stamps at the counter, and collect packages that could not be delivered to their home address."),
 ("药房", "pharmacy",  "Prescriptions are dispensed here by trained staff, customers buy painkillers and cold remedies, and advice is given on dosage and side effects."),
 ("超市", "supermarket","Shoppers push trolleys along aisles here selecting groceries, and cashiers scan items at the checkout before customers pack their bags."),
 ("车站", "station",   "Commuters wait on platforms here for trains to arrive, announcements list departure times, and tickets are checked at the barriers."),
 ("公园", "park",      "Families walk along paths here among trees and flower beds, children use the playground equipment, and joggers exercise on the grass in the morning."),
 ("博物馆", "museum",  "Exhibits and historical artefacts are displayed here in glass cases, visitors follow guided tours, and curators care for the permanent collection."),
 ("教堂", "church",    "Congregations gather here for services and hymns, weddings and funerals are held in the main hall, and the building has stained glass windows."),
 ("体育馆", "gymnasium","Athletes train here on equipment and weights, teams practise indoor sports on the court, and members attend fitness classes in the evenings."),
 ("剧院", "theatre",   "Audiences watch live performances here from tiered seating, actors rehearse on stage before opening night, and tickets sell out for popular plays."),
 ("大学", "university","Undergraduates attend lectures here and sit final examinations, researchers publish academic papers, and degrees are awarded at graduation ceremonies."),
 ("监狱", "prison",    "Inmates are held here in cells under supervision, guards patrol the corridors and wings, and visitors are searched before entering the facility."),
 ("码头", "harbour",   "Cargo ships dock here to unload containers, cranes lift freight onto the quayside, and fishing boats moor overnight alongside the pier."),
]

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base",
          "BAAI/bge-m3"]

DOCS = [d for _, _, d in ITEMS]
N = len(ITEMS)
print(f"{N} entities, corpus = {N} English documents, chance = {1/N:.3f}")

# which Chinese words SHARE a token-id sequence -> mathematically tied
tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
idmap = {}
for zh, en, _ in ITEMS:
    idmap.setdefault(tuple(tk(zh, add_special_tokens=False)["input_ids"]), []).append(en)
tied = {k: v for k, v in idmap.items() if len(v) > 1}
print("\nTOKEN-ID COLLISION GROUPS (all-MiniLM-L6-v2 / 30522 vocab):")
for k, v in tied.items():
    print(f"  ids={list(k)} -> {len(v)} entities SHARE ONE VECTOR: {', '.join(v)}")
n_tied = sum(len(v) for v in tied.values())
print(f"  {n_tied}/{N} entities live in a collision group")

dev = "cuda" if torch.cuda.is_available() else "cpu"
res = {}
for name in MODELS:
    m = SentenceTransformer(name, device=dev)
    short = name.split("/")[-1]
    D = m.encode(DOCS, normalize_embeddings=True, show_progress_bar=False,
                 batch_size=16)
    for mode in ("chinese", "english_control"):
        qs = [zh if mode == "chinese" else en for zh, en, _ in ITEMS]
        Q = m.encode(qs, normalize_embeddings=True, show_progress_bar=False,
                     batch_size=16)
        top1 = (Q @ D.T).argmax(1)
        hits = [int(t == i) for i, t in enumerate(top1)]
        acc = float(np.mean(hits))
        # accuracy restricted to entities inside a collision group
        tied_en = {e for v in tied.values() for e in v}
        ti = [i for i, (_, en, _) in enumerate(ITEMS) if en in tied_en]
        tacc = float(np.mean([hits[i] for i in ti])) if ti else float("nan")
        res[f"{short}|{mode}"] = {"top1": round(acc, 4),
                                  "collision_subset_top1": round(tacc, 4),
                                  "hits": int(sum(hits)), "n": N}
        print(f"  {short:20} {mode:16} top1={acc:.3f} ({sum(hits)}/{N})  "
              f"collision_subset={tacc:.3f}", flush=True)
        json.dump({"n": N, "tied_groups": {str(list(k)): v for k, v in tied.items()},
                   "acc": res, "complete": False}, open(OUT, "w"), indent=1)
    del m
    torch.cuda.empty_cache()

json.dump({"n": N, "tied_groups": {str(list(k)): v for k, v in tied.items()},
           "acc": res, "complete": True}, open(OUT, "w"), indent=1)
print("\n" + "=" * 70)
print("CROSS-LINGUAL RETRIEVAL: Chinese query -> English document corpus")
print(f"chance = {1/N:.3f}")
for k, v in res.items():
    print(f"  {k:38} top1={v['top1']:.3f}  collision_subset={v['collision_subset_top1']:.3f}")
print(f"wrote {OUT}")
