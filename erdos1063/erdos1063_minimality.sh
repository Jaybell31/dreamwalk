#!/usr/bin/env bash
# Erdos #1063 minimality sweep, k=60..75.
# For each k: exhaustively sweep the forced progressions for n < n_k.
# Expected verdict per k: NO_SOLUTION_BELOW_LIMIT  => n_k is MINIMAL (proven).
# Any HIT => the candidate is NOT minimal and the term is WRONG (that is the kill).
#
# Runs niced at low parallelism: the arena (night_shift_v3.sh) is PRIMARY on this box.
set -u
# Resolve to THIS script's directory so a fresh clone reproduces the sweep.
# Was `cd /home/jason/roundtable`, which silently sweeps the wrong tree (or
# fails) on any other machine -- unacceptable for the file that IS the
# minimality evidence.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
OUT=erdos1063_minimality.jsonl

# The sweeper is C and is NOT committed as a binary. Build it if missing;
# otherwise the loop below runs `./erdos1063_sweep` 16 times, fails silently
# into an empty JSONL, and the sweep "completes" having tested nothing.
if [ ! -x ./erdos1063_sweep ]; then
  echo "building erdos1063_sweep from source"
  gcc -O2 -o erdos1063_sweep erdos1063_sweep.c || {
    echo "FATAL: build failed"; exit 1; }
fi
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

# Refuse to stamp SWEEP_COMPLETE unless all 16 verdicts are present and every
# one is NO_SOLUTION_BELOW_LIMIT. A trailing completion marker on a short or
# hit-bearing run is exactly the artifact that makes an unproven claim look
# audited.
got=$(grep -c NO_SOLUTION_BELOW_LIMIT "$OUT" || true)
hits=$(grep -c '"hit":[^0]' "$OUT" || true)
if [ "$got" -ne 16 ] || [ "$hits" -ne 0 ]; then
  echo "SWEEP_INCOMPLETE got=$got/16 hits=$hits -- MINIMALITY NOT ESTABLISHED" >> "$OUT"
  echo "FAILED: got=$got/16 clear verdicts, $hits hits"
  exit 1
fi
echo "SWEEP_COMPLETE $(date -Is)" >> "$OUT"
echo "OK: 16/16 NO_SOLUTION_BELOW_LIMIT, 0 hits"
