# How much is the MSA actually doing here?

Ernest's first suggestion (2026-09-03):

> I think that you can try to shrink the MSA blocks since all the MHC proteins
> will have similar MSA, it is not a very useful signal. MSA block is large and if
> you can shrink it, you might significantly reduce the memory usage.

Measured against the actual dataset rather than argued about. Every number below
comes from `data/msa/*.npz` and `data/processed/manifest.json`.

## 1. The peptide has no MSA at all

This is the headline, and it was not what I expected to find.

| chain kind | n sampled | min depth | median | p90 | max |
|---|---|---|---|---|---|
| MHC heavy | 93 | 8,738 | **11,172** | 11,560 | 12,059 |
| β2m | 20 | 2,102 | 5,261 | 5,673 | 5,742 |
| **peptide** | 181 | 1 | **1** | 1 | **5** |

**Every peptide MSA is depth 1** — the query sequence and nothing else. Out of 181
sampled, the deepest had five sequences.

That is not a surprise on reflection: an 8–11mer has no detectable homologs. But
it has a sharp consequence. The MSA module cannot be contributing anything to the
peptide, which is the part of the complex this project exists to predict.

## 2. The heavy-chain MSA is *not* internally redundant

Ernest's stated reason for shrinking was redundancy. Measured on the first 2048
rows — the ones `max_seqs` actually keeps:

| msa_id | length | rows | unique | ≥90% id to query | ≥80% id | mean id |
|---|---|---|---|---|---|---|
| 6avf_e | 362 | 2048 | 2048 | 3 | 198 | 0.650 |
| 6k7t_a | 277 | 2048 | 2048 | 2 | 15 | 0.624 |
| 1qsf_a | 274 | 2048 | 2047 | 15 | 329 | 0.680 |
| 4pgb_a | 304 | 2048 | 2048 | 5 | 18 | 0.585 |
| 7duu_a | 273 | 2048 | 2048 | 10 | 440 | 0.691 |
| 6vrn_a | 293 | 2048 | 2048 | 3 | 47 | 0.633 |

Essentially every row is unique, mean identity to the query is 58–69%, and only a
handful of sequences sit above 90%. The search pulls in the MHC-I superfamily
across species, not a pile of near-identical HLA alleles.

So `max_seqs: 2048` is **truncating** a ~11,000-sequence alignment down to 2048
genuinely distinct sequences. It is not trimming padding, the way `max_atoms:
4608 → 3904` was.

## 3. But the MSAs *are* nearly the same from sample to sample

There is a more charitable reading of the suggestion, and it holds. Aligned
identity between the query sequences of *different samples*:

| | pairs | median | p10 | p90 |
|---|---|---|---|---|
| MHC heavy chain | 300 | **0.766** | 0.647 | 0.896 |
| peptide | 300 | **0.211** | 0.083 | 0.333 |

MHC heavy chains are ~77% identical to each other across the dataset. So while
each individual MSA is internally rich, it is describing a fold that is nearly
constant across all 1,084 samples. It is informative in absolute terms and close
to non-discriminative *between* samples — which is, I think, what Ernest meant.

Peptides at 21% identity are the opposite: maximally variable, and with no MSA.

## 4. What this means for each suggestion

**Shrinking the MSA blocks — worth doing, for a different reason than stated.**

The parameter argument does not work: `msa_module` is 3.2 M of 432 M parameters,
under 1%, so shrinking it saves almost no weight memory. But two things do
support it:

* Every fatal OOM traceback in `reports/ft_*.log` goes through
  `MSAModule.forward`, never `PairformerModule`. The MSA module runs first in the
  trunk (`model.py:304`) and carries `m [1, 2048, 512, 64]` on top of `z`, so it
  is where the trunk's memory actually peaks.
* Section 3 says the signal is largely constant across samples.

So the lever is `data.max_seqs`, not `msa_blocks`. Going 2048 → 512 cuts the MSA
tensor 4x. Unlike `max_atoms`, this *does* discard real sequences, so it has to be
measured rather than assumed.

Boltz has a built-in ablation for the extreme case: `no_msa: bool` on
`Boltz1.__init__` (`model.py:72`, guarded at `:186` and `:311`). A
`validation_only` run with `model.no_msa=true` bounds the entire question — if
lDDT barely moves with the MSA removed completely, then `max_seqs` can be cut
hard and the memory follows.

Cheap experiment ladder, all `validation_only`, all comparable to the 0.8827
baseline because none of them touch `validation_args`:

| run | change | what it answers |
|---|---|---|
| A | `data.max_seqs=512` | does 4x less MSA cost anything? |
| B | `data.max_seqs=128` | how far can it go? |
| C | `model.no_msa=true` | what is the MSA worth at all? |

**Peptide templates — the measurement makes this argument much stronger.**

`reports/TEMPLATES.md` covers the blocker (Boltz-1 has no template pathway; Boltz-2
does). But section 1 sharpens *why* it is the right instinct: the peptide has an
MSA of depth 1. It is the single part of the complex where the model receives zero
evolutionary information, and it is simultaneously the most variable part
(21% identity across samples) and the thing being predicted.

A structural prior is not one option among several for the peptide. It is the only
channel through which extra information about the peptide could reach the model at
all. That is a better case for templates than "they might be helpful", and it is
worth putting to Ernest in those terms.

## Method note

Two measurement bugs were hit and fixed before these numbers were trusted, both
worth recording:

* Comparing sequences position-wise after truncating to the shorter length gave a
  median heavy-chain identity of 0.076. The sequences have different construct
  boundaries, so unaligned position-wise comparison is meaningless.
* `difflib.SequenceMatcher` then gave 0.011. Its `autojunk` heuristic treats any
  element appearing in more than 1% of a sequence longer than 200 as junk — with
  20 amino acids in a 280-residue chain, *every* residue qualifies and the
  comparison collapses. `autojunk=False` is required for sequence data.

Both produced confident, plausible-looking, wrong numbers. Decoding was verified
against `boltz.data.const.tokens` by eye first: the first residues of `1a1m_a`
decode to `GLY SER HIS SER MET ARG TYR PHE PHE THR`, i.e. GSHSMRYFFT, a textbook
HLA class-I N-terminus.
