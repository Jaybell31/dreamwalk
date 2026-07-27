/* erdos1063_sweep.c -- exhaustive MINIMALITY sweep for Erdos #1063 / OEIS A389360.
 *
 * n_k = least n >= 2k such that (n-i) | C(n,k) for exactly all but ONE 0<=i<k.
 *
 * COVERAGE ARGUMENT (this is the load-bearing part -- read it before trusting output):
 *   Let e be the unique failing index of a solution n. rickyc's proved result
 *   (erdosproblems.com/forum/thread/1063, 26 Jun + 10 Jul 2026) gives, for every
 *   prime power p^a || k,
 *        v_p(n-e) >= a + v_p(C(k-1,e)) + floor(log_p max(e, k-1-e)).
 *   Hence D(k,e) := prod_{p^a||k} p^(that exponent)  divides  n-e.
 *   Since k | D(k,e), also e == n mod k.
 *   So EVERY solution with failing index r lies on the progression n = r + D(k,r)*t.
 *   Sweeping r = 0..k-1 over that progression is therefore EXHAUSTIVE, not heuristic.
 *
 *   We do NOT take that theorem on faith: mode=control re-derives the PUBLISHED
 *   n_k (k=2..59, OEIS b-file) through this exact same sparse route. If D were
 *   wrong the sweep would skip real solutions and the control would MISMATCH.
 *
 * WHEEL (a necessary condition, used only to skip -- never to accept):
 *   For p with k/2 < p <= k we have v_p(k!) = 1, and (n-i)|C(n,k) requires
 *   sum_{j!=i} v_p(n-j) >= 1. Let m = n mod p; multiples of p among n-0..n-(k-1)
 *   sit at j = m and j = m+p only.
 *     m >= k        -> no multiple -> EVERY index fails      -> reject n
 *     m+p >= k      -> exactly one -> index m must be the one failure -> need m == r
 *     m+p <  k      -> two multiples -> p imposes nothing
 *
 * Every survivor is then checked against the FULL definition over all p <= k, and
 * accepted only if the bad-index set is exactly {r}.
 *
 * usage: ./erdos1063_sweep <k> <limit> <label>
 *   limit = exclusive upper bound on n. mode control: pass n_k+1 and expect a hit
 *   at exactly n_k. mode minimality: pass n_k and expect NO hit.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef unsigned long long u64;
#define MAXK 128
#define MAXP 64

static int primes[MAXP], nprimes;
static int ep[MAXP];          /* v_p(k!) */
static int lp[MAXP], nlarge;  /* indices into primes[] with k/2 < p <= k */

static void sieve(int k){
    nprimes = 0;
    for (int i = 2; i <= k; i++){
        int ip = 1;
        for (int j = 2; (long)j*j <= i; j++) if (i % j == 0){ ip = 0; break; }
        if (ip) primes[nprimes++] = i;
    }
}
static int legendre(int n, int p){ int v = 0; long q = p; while (q <= n){ v += n/q; q *= p; } return v; }
static int vp_u64(u64 x, int p){ int v = 0; while (x % (u64)p == 0){ x /= (u64)p; v++; } return v; }

/* D(k,r) = prod_{p^a||k} p^(a + v_p(C(k-1,r)) + floor(log_p max(r,k-1-r))) */
static u64 forced_modulus(int k, int r, int *overflow){
    u64 D = 1; *overflow = 0;
    int rr = r > (k-1-r) ? r : (k-1-r);
    for (int t = 0; t < nprimes; t++){
        int p = primes[t];
        if (k % p) continue;
        int a = 0, kk = k; while (kk % p == 0){ kk /= p; a++; }
        int vb = legendre(k-1,p) - legendre(r,p) - legendre(k-1-r,p);   /* v_p(C(k-1,r)) */
        int u = 0; if (rr >= 1){ long q = p; while (q <= rr){ u++; q *= p; } }
        for (int e = 0; e < a + vb + u; e++){
            if (D > (u64)4e18 / (u64)p){ *overflow = 1; return 0; }
            D *= (u64)p;
        }
    }
    return D;
}

/* FULL definitional check: return number of bad indices, and the first one. */
static int full_check(u64 n, int k, int *first_bad){
    int bad = 0; *first_bad = -1;
    static int v[MAXP][MAXK];
    static int S[MAXP];
    for (int t = 0; t < nprimes; t++){
        int p = primes[t]; int s = 0;
        for (int j = 0; j < k; j++){ v[t][j] = vp_u64(n - (u64)j, p); s += v[t][j]; }
        S[t] = s;
    }
    for (int i = 0; i < k; i++){
        for (int t = 0; t < nprimes; t++){
            if (S[t] - v[t][i] < ep[t]){                /* (n-i) does NOT divide C(n,k) */
                if (++bad == 1) *first_bad = i;
                break;
            }
        }
        if (bad > 1) return bad;                        /* early out: >1 failure */
    }
    return bad;
}

int main(int argc, char **argv){
    if (argc < 4){ fprintf(stderr, "usage: %s k limit label\n", argv[0]); return 2; }
    int k = atoi(argv[1]);
    u64 limit = strtoull(argv[2], NULL, 10);
    const char *label = argv[3];
    if (k < 2 || k >= MAXK){ fprintf(stderr, "bad k\n"); return 2; }

    sieve(k);
    for (int t = 0; t < nprimes; t++) ep[t] = legendre(k, primes[t]);
    nlarge = 0;
    for (int t = 0; t < nprimes; t++) if (2*primes[t] > k) lp[nlarge++] = t;

    u64 tested = 0, wheel_pass = 0;
    u64 hit = 0; int hit_r = -1;
    clock_t t0 = clock();

    /* NOTE: we must take the MINIMUM over all r, not the first r that hits --
     * r-order is not n-order. Each r's progression is scanned in increasing n, so
     * we can shrink the per-r bound to the best n found so far and keep it exact. */
    u64 bound = limit;
    for (int r = 0; r < k; r++){
        int ov; u64 D = forced_modulus(k, r, &ov);
        if (ov || D == 0) continue;                     /* D > 4e18 >> limit: no candidate */
        if (D >= bound) continue;

        /* start at the first n on this progression with n >= max(2k, r+D), so the
         * incremental residue state below never needs a skip (a `continue` would
         * desynchronise it). */
        u64 n0 = (u64)r + D;
        if (n0 < (u64)(2*k)){
            u64 need = (u64)(2*k) - n0;
            n0 += ((need + D - 1) / D) * D;
        }

        /* incremental residues for the wheel primes */
        int res[MAXP], dmod[MAXP], pv[MAXP];
        for (int a = 0; a < nlarge; a++){
            int p = primes[lp[a]];
            pv[a]   = p;
            dmod[a] = (int)(D % (u64)p);
            res[a]  = (int)(n0 % (u64)p);
        }
        for (u64 n = n0; n < bound; n += D){
            tested++;
            int ok = 1;
            for (int a = 0; a < nlarge; a++){
                int m = res[a], p = pv[a];
                if (m >= k){ ok = 0; }                  /* zero multiples -> all fail */
                else if (m + p >= k && m != r){ ok = 0; }/* lone multiple not at r */
                if (!ok) break;
            }
            if (ok){
                wheel_pass++;
                int fb; int nbad = full_check(n, k, &fb);
                if (nbad == 1 && fb == r){
                    hit = n; hit_r = r;
                    bound = n;      /* only smaller n can win from later r */
                    break;          /* this r's progression is increasing: done */
                }
            }
            /* advance residues */
            for (int a = 0; a < nlarge; a++){
                res[a] += dmod[a];
                if (res[a] >= pv[a]) res[a] -= pv[a];
            }
        }
    }

    double secs = (double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("{\"label\":\"%s\",\"k\":%d,\"limit\":%llu,\"candidates_tested\":%llu,"
           "\"wheel_survivors\":%llu,\"hit\":%llu,\"hit_r\":%d,\"secs\":%.1f,"
           "\"verdict\":\"%s\"}\n",
           label, k, limit, tested, wheel_pass, hit, hit_r, secs,
           hit ? "HIT" : "NO_SOLUTION_BELOW_LIMIT");
    fflush(stdout);
    return 0;
}
