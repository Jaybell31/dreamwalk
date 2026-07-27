#!/usr/bin/env bash
# Erdos #1063 minimality sweep, k=60..75.
# For each k: exhaustively sweep the forced progressions for n < n_k.
# Expected verdict per k: NO_SOLUTION_BELOW_LIMIT  => n_k is MINIMAL (proven).
# Any HIT => the candidate is NOT minimal and the term is WRONG (that is the kill).
#
# Runs niced at low parallelism: the arena (night_shift_v3.sh) is PRIMARY on this box.
set -u
cd /home/jason/roundtable
OUT=erdos1063_minimality.jsonl
: > "$OUT"

declare -A N=(
 [60]=2117441088029 [61]=569181900 [62]=155275274056 [63]=216949362182
 [64]=2600001558 [65]=1936512990784 [66]=1618958347825 [67]=1178168706
 [68]=260686545186 [69]=709479687556 [70]=66383308992066 [71]=13760436550
 [72]=890473430071 [73]=40861892736 [74]=3146202243081 [75]=32639932074432
)

# cheapest k first so we bank results early
ORDER="67 64 71 73 61 62 63 68 74 60 72 69 66 65 75 70"

JOBS=3
for k in $ORDER; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  ( nice -n 19 ./erdos1063_sweep "$k" "${N[$k]}" "minimality_k$k" >> "$OUT" ) &
done
wait
echo "SWEEP_COMPLETE $(date -Is)" >> "$OUT"
