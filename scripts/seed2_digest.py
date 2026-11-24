# Does the Round 6 arm ordering survive a second, independent training run?
#
# Round 6 found abrupt / taper / b9-3e-4 separated by less than the draw-to-draw
# noise of a single eval (scripts/round6_noise.py). That cannot distinguish "the
# arms are equivalent" from "one arm is better and we measured it badly". A
# second training run of two arms can: if the ordering flips, the Round 6 gap
# was training noise; if it holds with the same sign in both runs, it is small
# but real.
#
# Everything is re-derived from the per-record jsonl rows, never copied from an
# earlier report (reports/ls6/RESULT_round6.md inherited a stale 0.02 A noise
# figure that way and it changed the conclusion).
#
#   python scripts/seed2_digest.py
#
# POOLING matches round6_noise.py exactly: a record's score is the mean over all
# 6 samples (3 per draw x 2 draws), so each seed carries the same draw noise.
# Mixing a 1-draw seed against a 2-draw seed would put all the noise on one side.
import json, glob, pathlib, statistics as S
from math import comb


def load(p):
    out = {}
    for f in glob.glob(p + "/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if not r.get("sanity"):
                out[r["record_id"]] = r
    if not out:
        raise SystemExit("!! no rows under %s -- did the eval finish?" % p)
    return out


def p2(k, n):
    """Two-sided sign test. k of n improved under H0: p=0.5."""
    if n == 0:
        return 1.0
    o = comb(n, k)
    return min(1.0, sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= o) / 2 ** n)


sub = set(pathlib.Path("data/processed/subset30_ids.txt").read_text().split())

# arm -> seed -> (draw1 dir, draw2 dir)
ARMS = {
    "abrupt": {
        1: ("runs/eval_val99/abrupt_last_d1", "runs/eval_val99/abrupt_d1_r2"),
        2: ("runs/eval_val99/abrupt_s2", "runs/eval_val99/abrupt_s2_r2"),
    },
    "b9_3e-4": {
        1: ("runs/eval_val99/b9_3e-4_last_d1", "runs/eval_val99/b9_3e-4_d1_r2"),
        2: ("runs/eval_val99/b9_s2", "runs/eval_val99/b9_s2_r2"),
    },
}

RAW, POOL = {}, {}
for arm, seeds in ARMS.items():
    for seed, (a, b) in seeds.items():
        A, B = load(a), load(b)
        ids = [i for i in A if i in B]
        RAW[(arm, seed)] = (A, B, ids)
        POOL[(arm, seed)] = {i: S.mean(A[i]["pep_rmsd"] + B[i]["pep_rmsd"]) for i in ids}


def line(tag, vals):
    v = sorted(vals)
    return "%-26s n=%-3d median %.3f  mean %.3f  p90 %.3f" % (
        tag, len(v), S.median(v), S.mean(v), v[int(0.9 * (len(v) - 1))])


print("=== ROW COUNTS (a seed missing rows means an eval did not finish) ===")
for k, (A, B, ids) in RAW.items():
    print("  %-14s seed %d: draw1 %d, draw2 %d, both %d" % (k[0], k[1], len(A), len(B), len(ids)))

print()
print("=== DRAW-TO-DRAW NOISE, per arm per seed (same checkpoint, two evals) ===")
print("    This is the yardstick every difference below has to clear.")
for k, (A, B, ids) in RAW.items():
    d = [abs(A[i]["pep_rmsd_mean"] - B[i]["pep_rmsd_mean"]) for i in ids]
    m1 = S.median([A[i]["pep_rmsd_mean"] for i in ids])
    m2 = S.median([B[i]["pep_rmsd_mean"] for i in ids])
    print("  %-14s seed %d: draw medians %.3f / %.3f (delta %+.3f)   |per-record diff| median %.3f"
          % (k[0], k[1], m1, m2, m2 - m1, S.median(d)))

print()
print("=== POOLED MEDIANS (6 samples/record) ===")
for arm in ARMS:
    for seed in (1, 2):
        P = POOL[(arm, seed)]
        ho = [i for i in P if i not in sub]
        print("  " + line("%s seed %d  all" % (arm, seed), list(P.values())))
        print("  " + line("%s seed %d  held-out" % (arm, seed), [P[i] for i in ho]))

print()
print("=== SEED EFFECT: does the same recipe land in the same place twice? ===")
for arm in ARMS:
    P1, P2 = POOL[(arm, 1)], POOL[(arm, 2)]
    ids = [i for i in P1 if i in P2]
    for nm, ss in [("all %d" % len(ids), ids), ("held-out %d" % len([i for i in ids if i not in sub]),
                                                [i for i in ids if i not in sub])]:
        diff = [P2[i] - P1[i] for i in ss]
        better = sum(x < 0 for x in diff)
        print("  %-10s seed2 - seed1  %-14s paired median %+.3f   seed2 better %2d/%2d  p=%.3f"
              % (arm, nm, S.median(diff), better, len(ss), p2(better, len(ss))))

print()
print("=== ARM EFFECT within each seed: abrupt vs b9_3e-4, both query-only ===")
print("    The Round 6 question. A negative median means abrupt is better.")
signs = {}
for seed in (1, 2):
    A, B = POOL[("abrupt", seed)], POOL[("b9_3e-4", seed)]
    ids = [i for i in A if i in B]
    for nm, ss in [("all %d" % len(ids), ids), ("held-out %d" % len([i for i in ids if i not in sub]),
                                                [i for i in ids if i not in sub])]:
        diff = [A[i] - B[i] for i in ss]
        better = sum(x < 0 for x in diff)
        med = S.median(diff)
        if nm.startswith("held-out"):
            signs[seed] = med
        print("  seed %d  abrupt - b9  %-14s paired median %+.3f   abrupt better %2d/%2d  p=%.3f"
              % (seed, nm, med, better, len(ss), p2(better, len(ss))))

print()
print("=== VERDICT INPUTS ===")
if len(signs) == 2:
    s1, s2 = signs[1], signs[2]
    agree = (s1 < 0) == (s2 < 0)
    print("  held-out abrupt-minus-b9: seed 1 %+.3f, seed 2 %+.3f -> sign %s"
          % (s1, s2, "AGREES" if agree else "FLIPS"))
    noise = S.median([S.median([abs(RAW[k][0][i]["pep_rmsd_mean"] - RAW[k][1][i]["pep_rmsd_mean"])
                                for i in RAW[k][2]]) for k in RAW])
    print("  typical single-eval per-record draw noise: %.3f A" % noise)
    print("  |seed 1 effect| %.3f, |seed 2 effect| %.3f -- compare against the noise above."
          % (abs(s1), abs(s2)))
    print("  A sign that FLIPS, or an effect under the noise in either seed, means the")
    print("  arms are not separated by this experiment. Say that plainly if so.")
