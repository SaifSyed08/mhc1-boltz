# Defects found in upstream Boltz v2.2.1

Audit run 2026-11-17 against a clean clone of `jwohlwend/boltz` at `v2.2.1`
(HEAD `b1ebfc4`, 107 Python files). Two agents searched; **every claim below was
then re-verified by hand at the cited line**, and the ones that did not survive
that check are listed at the bottom.

Round 2 (2026-11-21) added two more confirmed defects and filed eight items
upstream. See "Round 2" and "Filed upstream" at the bottom.

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

`data/feature/featurizer.py:210` and `featurizerv2.py:324` (re-checked 2026-11-21 —
the v2 line is 324, not 210 as first written here):

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

---

# Round 2 — 2026-11-21

Four agents, scoped to regions round 1 never touched: the inference/output path;
affinity + steering potentials; input parsing and constraints; losses, confidence,
cropping and tokenisation. Same evidence contract (file:line, quoted code, concrete
failure scenario, CONFIRMED vs PLAUSIBLE). Every claim below was re-verified by hand
before filing. Line numbers still against `b1ebfc4`, which is also `origin/main`.

## G. Chains that differ only in `modifications` or `cyclic` are silently merged

**Severity: high. User-facing, documented input syntax, silent wrong answer.**

`data/parse/schema.py:1037`. The comment above the loop says items are grouped by
"type, sequence and modifications"; the key is `(entity_type, seq)`. Each group is
then read from `items[0]` only — modifications at `:1168`, `cyclic` at `:1178` — and
the one resulting `parsed_chain` is assigned to every id in the group at
`:1301-1308`.

```yaml
sequences:
  - protein: {id: A, sequence: MKTAYIAK}
  - protein: {id: B, sequence: MKTAYIAK, modifications: [{position: 3, ccd: SEP}]}
```

Both chains are built from the unmodified entry; B's SEP is dropped. Listing B
first gives *both* chains the SEP. `msa` has an explicit cross-item check for this
exact hazard at `:1117-1124`; `modifications` and `cyclic` have none.

Also at `:1168`: `idx = mod["position"] - 1` with no bounds check, so `position: 0`
modifies the last residue of the chain instead of raising.

## H. `--max_parallel_samples` does not cap anything

**Severity: medium-high. Affects anyone hitting OOM at high `diffusion_samples`.**

`model/modules/diffusionv2.py:383` and `diffusion.py:541` — both versions:

```python
sample_ids_chunks = sample_ids.chunk(
    multiplicity % max_parallel_samples + 1
)
```

`chunk(n)` takes a *number of chunks*, not a chunk size. 6 samples with a cap of 3
gives one chunk of 6; 10 with a cap of 5 gives one chunk of 10; 10 with a cap of 3
gives two chunks of 5. The affinity path passes `max_parallel_samples: 1`
(`main.py:1376`) with `diffusion_samples_affinity` defaulting to 5, so that runs as
one chunk of 5 too. Predictions unchanged; only peak memory.

Relevant to us directly — this is the knob that was supposed to bound VRAM.

## Round 2 findings deliberately NOT filed

* **`com_counts` computed and discarded** in `potentials.py:111`, so the COM
  gradient is never divided by atoms per chain. Real-looking, but open PR
  [#718](https://github.com/jwohlwend/boltz/pull/718) already covers COM reduction
  and gradient scaling.
* **`--write_full_pae` help text contradicts the flag.** Covered by open PR
  [#602](https://github.com/jwohlwend/boltz/pull/602).
* **Contact potential underflow / kernel fallback.** Covered by open PR
  [#682](https://github.com/jwohlwend/boltz/pull/682).
* **`--subsample_msa` is a bare `is_flag`** (`main.py:1022`) whose help says
  "Default is True", while the dataclass default at `main.py:104` is `False` and the
  function signature default at `:1076` is `True`. Click always wins, so subsampling
  is off and cannot be turned off. Verified, but it is a three-way documentation
  disagreement where the intended default is genuinely unclear, so it is a question
  for the maintainers, not a patch. Not filed.
* **`confidencev2.py:489-495` fallback builds `pair_chains_iptm` as a tensor** where
  consumers expect a nested dict, and **`confidencev2.py:285` does not
  `repeat_interleave` `asym_id`** under `use_separate_heads`. Both PLAUSIBLE only —
  neither agent could produce a trigger, and `use_separate_heads` defaults to False.
* An agent reported that `--max_parallel_samples 5` yields "chunks of 5". That is
  wrong — see **H**. Caught by reading the line, which is the whole point of the
  verification pass.

Loss/confidence/cropping/tokenisation came back with **nothing airtight**, which is
a real result: that part of the tree is in better shape than the data pipeline.

---

# Filed upstream — 2026-11-21

Fork: `SaifSyed08/boltz`. Commits authored as `SaifSyed08 <saifsyed2999@gmail.com>`,
dated 2026-11-19, no AI attribution.

**Pull requests** (fix is unambiguous, nothing about trained weights changes):

| PR | Finding | Change |
|----|---------|--------|
| [#726](https://github.com/jwohlwend/boltz/pull/726) | F | three `_target_` lines in the shipped training configs |
| [#727](https://github.com/jwohlwend/boltz/pull/727) | H | `chunk()` -> `split()` in both diffusion samplers |
| [#728](https://github.com/jwohlwend/boltz/pull/728) | G | modifications + cyclic added to the entity grouping key |
| [#729](https://github.com/jwohlwend/boltz/pull/729) | A, E | unpack arity in `boltz2.py`; `ref_space_uid` off-by-one |

**Issues** (the fix is a maintainer judgement call, so reported not patched):

| Issue | Finding | Why not a PR |
|-------|---------|--------------|
| [#730](https://github.com/jwohlwend/boltz/issues/730) | B | if the weights were trained with the features dead, fixing it creates a train/inference mismatch — asked which way they want it |
| [#731](https://github.com/jwohlwend/boltz/issues/731) | new | failed input predicted as record 0; the right shape of the fix (sentinel vs re-raise) is theirs to pick |
| [#732](https://github.com/jwohlwend/boltz/issues/732) | C | present in v1 too, so the weights may depend on it |
| [#733](https://github.com/jwohlwend/boltz/issues/733) | D | design call: guard `chain_in_crop[j]` or drop masked chains earlier |

#733 states plainly that the 7ndt reproduction is from v1.0.0 and that the v2.2.1
path has not been re-executed. #730 and #732 say outright that we cannot tell from
source whether the released weights depend on the broken behaviour.

Every claim was checked against the 30+ open upstream PRs first; three candidates
were dropped as already-reported, and #731 explicitly notes that PR #720 addresses a
different failure path.

The four unverified leads from round 1 are still unfiled and still quarantined.
