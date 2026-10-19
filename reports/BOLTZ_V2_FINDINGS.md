# Defects found in upstream Boltz v2.2.1

Audit run 2026-11-17 against a clean clone of `jwohlwend/boltz` at `v2.2.1`
(HEAD `b1ebfc4`, 107 Python files). Two agents searched; **every claim below was
then re-verified by hand at the cited line**, and the ones that did not survive
that check are listed at the bottom.

Nothing here has been reported upstream yet. Saif decides what gets filed.

Line numbers are in `src/boltz/` unless a `scripts/` path is given.

---

## A. Boltz-2 confidence training silently trains on nothing when `symmetry_correction` is on

**Severity: high. Fully traced, deterministic, needs no data to reproduce.**

`data/feature/symmetry.py:377` — `minimum_lddt_symmetry_coords` returns **three**
values (confirmed by AST, not by eye):

```
return best_true_coords, best_rmsd, best_true_resolved_mask.unsqueeze(0)
```

`model/models/boltz2.py:749`, inside `get_true_coordinates` (lines 725–792),
unpacks **two**:

```python
best_true_coords, best_true_coords_resolved_mask = (
    minimum_lddt_symmetry_coords(
        coords=out["sample_atom_coords"][i : i + 1],
        feats=batch,
        index_batch=idx,
    )
)
```

That raises `ValueError: too many values to unpack` on every call. The caller is
`model/models/boltz2.py:852`, inside `training_step`, wrapped in:

```python
try:            # lines 850-860
    ...
except Exception as e:
    print(f"Skipping batch with id {batch['pdb_id']} due to error: {e}")
```

So every batch raises, prints one line, and is skipped. **Boltz-2 confidence
training with `symmetry_correction=True` processes zero batches** and reports no
error beyond a print.

`model/models/boltz1.py:423` unpacks all three correctly, so this is a Boltz-2-only
regression.

**Fix:** unpack three values at `boltz2.py:749`, discarding the rmsd.

---

## B. MSA deletion features are always zero, in both featurizers

**Severity: high. Affects every prediction and every training run.**

`data/feature/featurizer.py:313-323` and, identically,
`data/feature/featurizerv2.py:427-437`:

```python
for chain_id, chain_msa in msa.items():
    chain_deletions = chain_msa.deletions
    for sequence in chain_msa.sequences:
        seq_idx = sequence["seq_idx"]
        del_start = sequence["del_start"]
        del_end = sequence["del_end"]
        chain_deletions = chain_deletions[del_start:del_end]   # <-- overwrites itself
```

The loop variable is reassigned from its own slice, so each iteration slices the
*previous* slice using offsets that index the original array.

The first sequence in an MSA is the query, which by definition has no insertions
relative to itself. `data/parse/a3m.py:84-85` (and `data/parse/csv.py:83-84`) set:

```python
del_start = len(deletions)
del_end = del_start + len(deletion)
```

so for the query `del_start == del_end == 0`. After the first iteration
`chain_deletions` is `deletions[0:0]`, i.e. empty, and every later slice of an
empty array is empty. **The `deletions` dict is therefore never populated**, and
`has_deletion`, `deletion_value` and `deletion_mean` are zero for every sequence
of every MSA — training, validation and inference alike.

**Fix:** slice fresh each iteration — `chain_msa.deletions[del_start:del_end]`.

**Open question we cannot answer from the source:** whether the released weights
were trained with these features already dead. If they were, the features are
merely inert; if they were not, there is a train/inference mismatch. Do not assert
either in a bug report.

---

## C. The `visited` set that should exclude already-paired sequences never matches

**Severity: medium. Produces duplicate MSA rows that displace distinct sequences.**

`data/feature/featurizer.py:210` and the same line in `featurizerv2.py`:

```python
taxonomy_map = sorted(taxonomy_map.items(), key=..., reverse=True)
# -> list of (taxon, [(chain_id, seq_idx), ...])
visited = {(c, s) for c, items in taxonomy_map for s in items}
...
    i for i in range(1, len(msa[c].sequences)) if (c, i) not in visited
```

The comprehension binds `c` to the **taxon** and `s` to a `(chain_id, seq_idx)`
**tuple**, so `visited` holds `(taxon, (chain_id, seq_idx))`. The membership test
`(c, i)` — a chain id and an `int` — can never match.

So `available[c]` keeps every sequence, including ones already consumed as
taxonomy-paired rows. Those are emitted a second time as unpaired rows, and after
`pairing[:max_seqs]` truncation the duplicates displace distinct sequences.

**Fix:** `{(c, s) for _, items in taxonomy_map for (c, s) in items}`.

Present in v1.0 too, so the released weights may have been trained with it. Claim
only that the code does not do what it says.

---

## D. Symmetry correction can swap a kept chain for a dataset-masked copy

**This is Saif's own v1.0.0 finding (`UPSTREAM_PATCHES.md` §8), still present.**

`data/feature/symmetry.py:460-473`:

```python
for i, chain in enumerate(structure.chains):
    ...
    if chain_in_crop[i]:
        possible_swaps = []
        for j, chain2 in enumerate(structure.chains):
            ...
            if (chain["entity_id"] == chain2["entity_id"]
                    and end - start == end2 - start2):
                possible_swaps.append((start, end, start2, end2, i, j))
```

`chain_in_crop[i]` guards the chain being replaced; **`chain_in_crop[j]` is never
checked for the replacement**. The same pattern is at `data/mol.py:559-570`, the
Boltz-2 path.

Masked chains survive into `structure.chains` (`data/module/training.py:130` passes
`mask=structure["mask"]`) while the tokenizer drops them
(`data/tokenize/boltz.py:82`: `chains = struct.chains[struct.mask]`). So a masked
chain is invisible to tokenisation yet remains a legal swap target.

Observed in v1.0.0 on PDB `7ndt`: a prediction 0.5 Å from the truth scored 48.9 Å.

**Fix:** require `chain_in_crop[j]` as well, or drop masked chains before validation.

---

## E. Off-by-one: the last reference conformer is never augmented (Boltz-2)

`data/feature/featurizerv2.py:1495`:

```python
for i in range(torch.max(ref_space_uid)):
```

`range(max)` excludes `max`, so the highest `ref_space_uid` never receives the
random roto-translation. For a single-residue input (all atoms uid 0) nothing is
augmented at all. For protein + one ligand, the ligand is typically last, so its
`ref_pos` is always fed in the raw RDKit/CCD frame.

**Fix:** `range(int(torch.max(ref_space_uid)) + 1)`.

Low impact, trivially correct, cheap to file.

---

## F. The shipped training configs point at a module that does not exist

`scripts/train/configs/structure.yaml:78`, `full.yaml:79`, `confidence.yaml:80`:

```yaml
_target_: boltz.model.model.Boltz1
```

There is no `src/boltz/model/model.py`. The class lives at
`src/boltz/model/models/boltz1.py:40`; `src/boltz/model/__init__.py` is empty, and
nothing in the tree imports `boltz.model.model`. Hydra cannot instantiate it.

One-line fix in three files. Note the pattern: Boltz's documented training entry
point was also broken as shipped in v1.0.0, in a different way.

---

## Verified as FIXED since v1.0.0 (no action)

* `chain_constraint_features` `UnboundLocalError` — now initialised to `{}` before
  the branch (`featurizer.py:1212`, `featurizerv2.py:2330`).
* `AlphaFoldLRScheduler` passing `verbose=` — gone (`model/optim/scheduler.py:71`).
* `center_random_augmentation` `NameError` — now imported
  (`model/modules/diffusion.py:19`).

## Reported by the agents but NOT independently verified here

Treat as leads, not findings. Do not file without checking.

* **Truncated symmetry combinations.** `symmetry.py:474-484` caps candidates with
  `itertools.islice(product, max_n_symmetries * 10)`. Because `product` varies the
  last factor fastest, the kept prefix is biased, and the agent's stdlib replica
  suggests the identity assignment can be excluded entirely for ≥5 same-entity
  chains. The code is as quoted; the consequence table is the agent's simulation,
  not a run of Boltz.
* **Ligand symmetry picks the last improving permutation, not the best.**
  `symmetry.py:291-352`; each candidate is compared against the unchanged base
  rather than the running best. Not traced by hand.
* **Validation substitutes record 0 on any failure.** `data/module/training.py:418,
  425, 438, 471`. Already known in v1.0.0; still present. Real but low novelty.
* **v1 pocket conditioning does not filter by `is_present`** (`featurizer.py` ~596).
  Marked PLAUSIBLE by the agent.

## Filing guidance

File separately, smallest first, each with the quoted lines and the one-line fix.
**A** and **F** are the most provable and need no data. **B** is the most
consequential and the most likely to draw a substantive reply. **D** carries Saif's
own reproduction from v1.0.0 and should say so plainly — the logic is unchanged in
v2.2.1, but it has not been re-executed there.
