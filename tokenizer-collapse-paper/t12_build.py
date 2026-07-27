#!/usr/bin/env python3
"""T12 -- POWERED semantic polarity test. Template-generated, n=96 pairs.

WHY NOT THE LOCAL MODEL: qwen3.5:35b was the intended miner, but the GPU is
owned by Q's arena night_shift job and the ollama daemon wedged with glm stuck
in "Stopping..." holding 22GB. Generation was unavailable. The local model was
a CONVENIENCE for drafting text, never the evidence -- so we template the items
instead and keep the same MECHANICAL validator. Nothing about the test's
validity depended on who wrote the sentences.

DESIGN (identical logic to T11, powered up):
  query  = "<carrier> <emoji>"   -- polarity carried ONLY by the emoji
  corpus = 2*N EMOJI-FREE prose docs, one bad + one good per item
  gold   = a DOCUMENT, not an emoji  -> not tautological
  demojize cannot leak: validator rejects any doc containing a demojized word.

PRE-REGISTERED DECISION RULE (fixed before looking at results):
  primary metric = pair-restricted 2-way accuracy, chance = 0.500
  H1: raw is at chance for collapsed models          (expected: forced 0.500)
  H2: demojize > raw, two-sided exact binomial, alpha = 0.05
      -> if H2 SIGNIFICANT: keep the "one-line fix restores retrieval" claim.
      -> if NOT: paper narrows to COLLAPSE + TIE ARTIFACT, and demojize is
         reported as INSUFFICIENT for semantic retrieval. No fishing.
  word-control must beat raw for every model, else the corpus is void.
"""
import json, itertools, re

OUT = "/home/jason/dream_harness/t12_pairs.json"

# 24 event templates x 4 domain instantiations = 96 items.
# Each: (carrier, bad_doc, good_doc). All docs emoji-free professional prose.
TEMPLATES = [
 ("the {thing} migration is finished",
  "Post-incident review: the {thing} migration left the primary replica inconsistent and we served corrupted records to customers for forty minutes before the rollback completed.",
  "Release note: the {thing} migration completed cleanly across the primary and every replica, with no downtime and no customer visible impact recorded."),
 ("we shipped the new {thing} flow",
  "Escalation summary: the new {thing} path double charged a subset of customers and the finance team is processing refunds while the release is held back.",
  "Launch summary: the new {thing} path is live for all customers, conversion improved week over week and no refunds have been required so far."),
 ("the quarterly {thing} numbers are in",
  "Board memo: {thing} revenue came in materially below plan for the third consecutive quarter and we are opening a formal review of the cost base.",
  "Board memo: {thing} revenue beat plan for the third consecutive quarter and we are accelerating hiring against the newly raised internal targets."),
 ("my {thing} review came back",
  "Engineering note: the {thing} change was rejected in review, the author was asked to redesign the approach and the branch has been closed unmerged.",
  "Engineering note: the {thing} change was approved in review with no requested edits and has already been merged into the mainline branch."),
 ("the {thing} client call just ended",
  "Account risk: the {thing} client used the call to formally raise a complaint about repeatedly missed deadlines and asked about exiting the contract early.",
  "Account update: the {thing} client used the call to confirm they are renewing and expanding their contract with us for another full year."),
 ("results from the {thing} load test",
  "Reliability report: the {thing} service fell over at forty percent of target load, latency exceeded the error budget and we paged the on call twice.",
  "Reliability report: the {thing} service held target load comfortably with latency well inside the error budget and no alerts were raised overnight."),
 ("the {thing} security audit landed",
  "Security finding: auditors identified a critical unauthenticated data exposure in {thing} and we are treating it as a live incident with mandatory disclosure.",
  "Security finding: auditors closed the {thing} engagement with no critical or high findings and confirmed our controls comfortably meet the standard."),
 ("update on the {thing} hiring round",
  "Recruiting update: the final {thing} candidate withdrew after the offer stage and the role is being reopened from scratch early next quarter.",
  "Recruiting update: the final {thing} candidate signed the offer and starts next month, closing the role several weeks ahead of schedule."),
 ("status of the {thing} backfill",
  "Data incident: the {thing} backfill wrote duplicate records into the warehouse and every downstream dashboard has been reporting inflated numbers since Tuesday.",
  "Data update: the {thing} backfill completed and reconciled exactly against source counts, so all downstream dashboards are now reporting accurate figures."),
 ("feedback from the {thing} beta users",
  "Product signal: beta users found the {thing} workflow confusing and slower than the tool it replaces, and most stopped using it within a week.",
  "Product signal: beta users found the {thing} workflow clearer and faster than the tool it replaces, and weekly retention is unusually strong."),
 ("the {thing} deployment to the eu region",
  "Ops incident: the {thing} regional rollout tripped a cascading failure across availability zones and we invoked the disaster recovery runbook at midnight.",
  "Ops update: the {thing} regional rollout finished zone by zone with health checks passing throughout and no rollback was required at any point."),
 ("news about the {thing} funding round",
  "Investor update: the lead investor withdrew from the {thing} round at the final stage and we are cutting burn to extend runway while we reassess.",
  "Investor update: the {thing} round closed oversubscribed with the lead confirmed and we are now funded well beyond the original operating plan."),
 ("the {thing} inspection report arrived",
  "Compliance notice: inspectors documented repeated deviations in {thing} handling and issued a corrective action order with a thirty day deadline.",
  "Compliance notice: inspectors documented no deviations in {thing} handling and renewed our certification for the maximum available period."),
 ("the {thing} contract negotiation wrapped up",
  "Legal summary: the counterparty walked away from the {thing} terms late in the process and we have no agreement in place for the coming period.",
  "Legal summary: the counterparty accepted the {thing} terms with only minor drafting changes and the agreement is signed and effective immediately."),
 ("the {thing} shipment status updated",
  "Logistics alert: the {thing} shipment was impounded at the border over documentation errors and customers should expect several weeks of delay.",
  "Logistics update: the {thing} shipment cleared customs without inspection and is arriving at the distribution centre ahead of the promised date."),
 ("the {thing} campaign results came through",
  "Marketing review: the {thing} campaign spent the full budget while generating almost no qualified pipeline, and we are pausing all remaining placements.",
  "Marketing review: the {thing} campaign came in under budget while generating record qualified pipeline, and we are extending it into the next quarter."),
 ("the {thing} trial enrolment update",
  "Study alert: {thing} enrolment stalled far below target and the sponsor is considering terminating the site for failing to meet its commitments.",
  "Study update: {thing} enrolment reached target ahead of schedule and the sponsor has asked whether the site can take an additional allocation."),
 ("the {thing} ticket queue this morning",
  "Support escalation: the {thing} queue tripled overnight after a failed release and our first response time is now far outside the agreed service level.",
  "Support update: the {thing} queue cleared overnight after the fix went out and first response time is now comfortably inside the agreed service level."),
 ("the {thing} pipeline forecast for the quarter",
  "Sales review: the {thing} forecast collapsed after two anchor deals slipped out of the quarter and we are now well short of the committed number.",
  "Sales review: the {thing} forecast strengthened after two anchor deals closed early and we are now comfortably ahead of the committed number."),
 ("the {thing} line ran overnight",
  "Manufacturing incident: the {thing} line produced an entire shift of out of tolerance parts and the batch has been quarantined pending disposition.",
  "Manufacturing update: the {thing} line produced a full shift within tolerance and the batch has been released straight to finished goods."),
 ("the {thing} model evaluation finished",
  "Modelling note: the {thing} model failed its holdout evaluation and performed worse than the existing baseline, so the rollout has been cancelled.",
  "Modelling note: the {thing} model passed its holdout evaluation and clearly outperformed the existing baseline, so the rollout has been approved."),
 ("the {thing} budget decision came down",
  "Finance memo: the {thing} budget request was denied and the existing allocation is being reduced, so planned work will have to be descoped.",
  "Finance memo: the {thing} budget request was approved in full with an additional contingency, so planned work can proceed at the intended scope."),
 ("the {thing} outage postmortem is done",
  "Postmortem: the {thing} outage lasted six hours, breached our published availability commitment and triggered contractual credits for major accounts.",
  "Postmortem: the {thing} outage was contained within nine minutes, stayed inside our published availability commitment and triggered no contractual credits."),
 ("the {thing} handover completed today",
  "Transition risk: the {thing} handover left critical runbooks undocumented and the incoming team cannot currently support the system without help.",
  "Transition update: the {thing} handover left every runbook documented and the incoming team is already supporting the system without assistance."),
]

THINGS = ["billing", "inventory", "onboarding", "reporting"]

EMOJI_PAIRS = [
    ("\U0001F480", "\U0001F680"), ("\U0001F4A9", "\U0001F389"),
    ("\U0001F4C9", "\U0001F4C8"), ("\U0001F62D", "\U0001F60E"),
    ("\U0001F621", "\U0001F91D"), ("\U0001F525", "\u2705"),
    ("\U0001F6A8", "\U0001F510"), ("\u274C", "\U0001F31F"),
    ("\U0001F4A5", "\U0001F44D"), ("\U0001F44E", "\U0001F44F"),
    ("\u26A0", "\U0001F7E2"), ("\U0001F614", "\U0001F600"),
]

BANNED = {"skull","rocket","fire","thumbs","checkmark","siren","poop","party",
          "chart","warning","emoji","face","star","clap","lock","handshake",
          "green","tada","confetti","cross","mark","up","down","trend"}

def has_emoji(s):
    return any(ord(c) > 0x2500 for c in s)

items, seen = [], set()
for i, ((carrier_t, bad_t, good_t), thing) in enumerate(
        itertools.product(TEMPLATES, THINGS)):
    carrier = carrier_t.format(thing=thing)
    bad = bad_t.format(thing=thing)
    good = good_t.format(thing=thing)
    # ---- MECHANICAL VALIDATION (same rules as the DF-miner version) ----
    if has_emoji(carrier) or has_emoji(bad) or has_emoji(good):
        continue
    if len(bad.split()) < 12 or len(good.split()) < 12:
        continue
    if bad.lower() == good.lower() or carrier.lower() in seen:
        continue
    words = set(re.findall(r"[a-z]+", (bad + " " + good).lower()))
    leak = words & BANNED
    if leak:
        print(f"  LEAK REJECT {carrier!r}: {sorted(leak)}")
        continue
    seen.add(carrier.lower())
    be, ge = EMOJI_PAIRS[len(items) % len(EMOJI_PAIRS)]
    items.append({"carrier": carrier, "bad_doc": bad, "good_doc": good,
                  "bad_emoji": be, "good_emoji": ge, "thing": thing})

json.dump(items, open(OUT, "w"), indent=1)
print(f"validated pairs: {len(items)}  -> {OUT}")
print(f"corpus size: {2*len(items)} emoji-free docs")
