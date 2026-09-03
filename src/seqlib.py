"""Small sequence toolkit: BLOSUM62 Smith-Waterman + k-mer homology scoring.

No biopython in this environment, so these are self-contained. Smith-Waterman is
plain Python (only called on short/ambiguous pairs); homology screening of long
chains uses k-mer containment, which is far cheaper and sufficient to separate
MHC-I heavy chains from unrelated crystal partners.
"""
from collections import Counter

_B62_RAW = """
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_AA = "ARNDCQEGHILKMFPSTWYV"
BLOSUM62 = {}
for _line in _B62_RAW.strip().splitlines():
    _p = _line.split(); _r = _p[0]
    for _c, _v in zip(_AA, _p[1:]):
        BLOSUM62[(_r, _c)] = int(_v)

# Modified/non-standard residues decode to 'X'; score them neutrally rather than
# as mismatches, so a peptide like KILGXVFXV still aligns to KILGFVFTV.
for _c in _AA + "X":
    BLOSUM62[("X", _c)] = 0
    BLOSUM62[(_c, "X")] = 0


def wildcard_identity(a, b):
    """Identity over non-X positions for equal-length sequences; None if unusable."""
    if not a or not b or len(a) != len(b):
        return None
    cmpd = [(x, y) for x, y in zip(a, b) if x != "X" and y != "X"]
    if not cmpd:
        return None
    return sum(x == y for x, y in cmpd) / len(cmpd)


def sw_align(a, b, gap_open=-11, gap_ext=-1):
    """Smith-Waterman (Gotoh). Returns (score, identity, aligned_len)."""
    if not a or not b:
        return 0, 0.0, 0
    n, m = len(a), len(b)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    E = [[-10**6] * (m + 1) for _ in range(n + 1)]
    F = [[-10**6] * (m + 1) for _ in range(n + 1)]
    P = [[0] * (m + 1) for _ in range(n + 1)]  # 0 stop 1 diag 2 up 3 left
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            E[i][j] = max(E[i][j - 1] + gap_ext, H[i][j - 1] + gap_open)
            F[i][j] = max(F[i - 1][j] + gap_ext, H[i - 1][j] + gap_open)
            d = H[i - 1][j - 1] + BLOSUM62.get((ai, b[j - 1]), -4)
            v = max(0, d, F[i][j], E[i][j])
            H[i][j] = v
            P[i][j] = 0 if v == 0 else (1 if v == d else (2 if v == F[i][j] else 3))
            if v > best:
                best, bi, bj = v, i, j
    ident = alen = 0
    i, j = bi, bj
    while i > 0 and j > 0 and P[i][j]:
        p = P[i][j]
        if p == 1:
            alen += 1
            if a[i - 1] == b[j - 1]:
                ident += 1
            i -= 1; j -= 1
        elif p == 2:
            alen += 1; i -= 1
        else:
            alen += 1; j -= 1
    return best, (ident / alen if alen else 0.0), alen

def kmers(s, k=5):
    return {s[i:i + k] for i in range(max(0, len(s) - k + 1))}

def kmer_containment(seq, ref_kmers, k=5):
    """Fraction of seq's k-mers present in a reference k-mer pool."""
    ks = kmers(seq, k)
    if not ks:
        return 0.0
    return len(ks & ref_kmers) / len(ks)
