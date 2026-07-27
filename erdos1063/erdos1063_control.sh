#!/usr/bin/env bash
# POSITIVE CONTROL: re-derive all 58 PUBLISHED A389360 terms (k=2..59) through the
# exact same sparse code path used for the k=60..75 minimality sweep.
#
# This is the instrument that makes the null results meaningful. If D(k,r) were
# wrong, or the wheel skipped real candidates, known terms would be MISSED here.
# A null result from a search that cannot rediscover the literature is worthless.
set -u
cd "$(dirname "$0")"

[ -f b389360.txt ] || curl -s -A "Mozilla/5.0" https://oeis.org/A389360/b389360.txt -o b389360.txt
[ -x erdos1063_sweep ] || gcc -O3 -march=native -o erdos1063_sweep erdos1063_sweep.c

python3 - <<'EOF'
import subprocess, json
bf = {}
for line in open('b389360.txt'):
    p = line.split()
    if len(p) == 2: bf[int(p[0])] = int(p[1])
ok, bad, rows, tot = 0, [], {}, 0
for k in sorted(bf):
    want = bf[k]
    # limit = want+1 so a correct searcher must land on exactly `want`
    r = subprocess.run(['./erdos1063_sweep', str(k), str(want+1), 'ctrl'],
                       capture_output=True, text=True)
    d = json.loads(r.stdout); rows[k] = d; tot += d['candidates_tested']
    if d['hit'] == want: ok += 1
    else:
        bad.append((k, want, d['hit']))
        print(f"MISMATCH k={k} published={want} sweep={d['hit']}")
print(f"=== SPARSE-ROUTE POSITIVE CONTROL: {ok}/{len(bf)} reproduced, mismatches={len(bad)}")
print(f"candidates tested k=2..59: {tot:,}  vs brute {sum(bf.values()):,}")
print("VERDICT:", "GREEN" if not bad else "RED -- searcher is unsound, do not trust any null result")
json.dump(rows, open('erdos1063_control_sparse.json','w'), indent=1)
EOF
