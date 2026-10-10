# Round 6 plan: hard peptides, data quality, templates, MSA

Ernest's reply (2026-10-12) to the eval result (`reports/ls6/RESULT_eval99.md`).

He framed the task this way:
* MHC-peptide is a training project. Its value is the low diversity of the data, so
  a limited improvement is fine.
* Focus on the complex cases first: unusual peptide lengths.
* Inspect them in PyMOL.
* Sort the peptides into three groups: can improve, particularly difficult, and
  drop (low-quality PDB entries).
* Look at templates, the main theme of MHC-Fine. Options include templates per
  peptide length and NetMHCpan's stricter anchor positioning, used to build full
  or partial templates.
* MSA: dropping it at inference alone will cost accuracy. Train it away instead,
  giving the model fewer MSA rows each step until it runs on the query sequence
  alone. If quality holds without the MSA, that is a result.
* Not TCR yet. When MHC-peptide is done, present the findings to the group.

## 1. Peptide triage (laptop, started)

`src/peptide_quality.py` → `data/peptide_quality.json`. For all 1,084 records it
records:
* the entry's resolution, R-free, clashscore and % RSRZ outliers;
* for the selected peptide chain, per-residue RSCC, RSRZ, B-factor and occupancy,
  the unobserved residues, and any ligands covalently linked to the peptide.

These come from the wwPDB validation data through the RCSB API. 1,020 of the 1,084
records have density data; the other 64 have no structure factors deposited.

First pass (2026-10-12):

| split | peptide mean RSCC < 0.8 | covalent ligand on peptide | unobserved peptide residues | resolution > 3.2 Å |
|---|---|---|---|---|
| train (914) | 28 | 4 | 32 | 18 |
| validation (99) | 4 | 1 | 4 | 3 |
| test (71) | 0 | 0 | 2 | 4 |

**The worst case, `7wj3` (GAAL, 15 Å), is a lipopeptide.** Its N-terminal Gly is
covalently linked to a myristoyl group (MYR). Our pipeline masks that group as a
ligand, so Boltz is asked to place a bare 4-mer whose real position is set by a
C14 lipid it never sees. That makes `7wj3` a "drop", or a separate lipopeptide
class; it is not a model failure. There are 3 MYR-linked records across the
splits.

The other tail cases have well-fit peptides:
* `7t5m` (20-mer): RSCC 0.88–0.98 except its last residue; 3 residues unobserved.
* `8d5n` (16-mer): ≥ 0.84.
* `6zkx` (12-mer): ≥ 0.78.
* `7ndq` (9-mer): ≥ 0.89.
* `7ow5` (10-mer): ≥ 0.92.

The crystal is trustworthy for all of these, so they are genuine "difficult" or
"can improve" cases.

Validation records whose peptide fits its density poorly: `7ejm` (mean RSCC 0.78,
min 0.50), `7ejn` (0.68, min 0.38), `7s8r` (0.79), `7sih` (0.77). An RMSD measured
against a poorly resolved peptide says little about the model.

Next:
* Join the quality table with the per-record eval rows. `scripts/ls6_round5.sh`
  prints them.
* Write the three groups to `reports/TRIAGE_peptides.md`, with the rule used for
  each:
  * **drop:** covalent modification, peptide mean RSCC < 0.8, or resolution > 3.5 Å;
  * **difficult:** length ≤ 7 or ≥ 12, or a non-canonical binding mode;
  * **can improve:** the rest, where the RMSD is above 1.5 Å.
* Run the eval and training medians again without the "drop" group.

## 2. PyMOL

Validation saves metrics, not coordinates. To overlay a prediction on the crystal,
the tail cases need a run that writes structures. Options:
* teach `src/peptide_rmsd.py` to dump predicted coordinates as PDB for a list of
  records; or
* run `boltz predict` on them. That needs the MSAs as a3m or csv, and ours are
  processed `.npz`.

The first is less work and scores exactly what the eval scored. Until then,
`fetch 7wj3` in PyMOL plus `show sticks, chain C or resn MYR` shows the lipid.

## 3. Templates

**Boltz-1 v1.0.0 has no template input.** The model has no template module;
templates arrived with Boltz-2. So MHC-Fine's "templates are crucial" cannot be
copied directly. Three routes:

1. **Pocket constraints, which Boltz-1 does have** (`constraints: pocket` in the
   input schema, through `chain_constraint_features`). Use them for NetMHCpan-style
   anchors: tell the model the peptide binds the B and F pockets, using the HC
   residues NetMHCpan's pseudo-sequence uses. This is a partial template at
   inference, with no retraining. It says which HC residues the peptide touches,
   not which peptide residue touches them.
2. **Move to Boltz-2**, which takes templates. That means a new model and new
   training code. It is the bigger step, and the one closest to MHC-Fine.
3. **Templates as training data:** thread each hard peptide onto a same-length
   known structure (or a 9-mer core plus a bulge, as NetMHCpan aligns peptides of
   other lengths) and fine-tune on those.

Recommendation: try 1 first. It is cheap and tests the anchor idea directly on
the hard cases. Ask Ernest whether 2 is in scope.

## 4. MSA

* **4a. Inference ablation (job ready).** `scripts/ls6_eval_msa.slurm` runs
  pretrained Boltz-1 on all 99 records with the MSA cut to 1 row (query only), 16
  and 128 rows. Validation takes the first N rows, and row 0 is the query. Full
  depth (2048) is `runs/eval_val99/pretrained`. This gives the size of the drop
  Ernest expects, per peptide length.
* **Training already varies MSA depth.** Upstream's featurizer draws a depth
  uniformly from 1 to `max_seqs` (2048) for every training sample
  (`featurizer.py` ~1174). So Boltz-1 and every fine-tune here have already seen
  shallow MSAs, but depth ≤ 16 in only ~0.8% of draws. Gradual removal moves that
  distribution down; it is not a new mechanism.
* **4b. Gradual removal (next).** A callback lowers `max_seqs` each epoch, e.g.
  2048 → 512 → 128 → 32 → 8 → 1, then trains on the query alone. The DataLoader
  does not keep workers between epochs (`persistent_workers` is unset), so each
  epoch's workers pick up the new value. Validation runs at depth 1 throughout and
  is compared with 4a's depth-1 result.

  A pMHC-specific point: the heavy chain and β2m have deep MSAs, but the peptide's
  MSA is essentially empty. What the MSA carries here is the MHC fold, which a
  model trained on 914 MHC structures may learn without it.

## 5. Running already (Round 5)

The batch-9 and batch-129 learning-rate arms. Their results are read with
`scripts/ls6_round5.sh` and will be written up in `reports/ls6/RESULT_round5.md`.

## 6. Built (2026-10-12), submitted with `scripts/ls6_submit_round6.sh`

| piece | code | job |
|---|---|---|
| Structures for PyMOL | `src/structure_dump.py` (env `STRUCTURE_DUMP_IDS`): `truth.cif`, `sample{k}.cif`, `view.pml` per record, superposed on the groove | inside the pocket eval |
| Pocket conditioning, NetMHCpan pseudo-sequence | `src/make_pockets.py` → `data/pockets_pseudoseq.json`. The 34 positions are mapped by alignment to HLA-A*02:01; the reference reproduces NetMHCpan's A*02:01 pseudo-sequence exactly. A median of 30 of the 34 lie within 6 Å of the crystal peptide. | `scripts/ls6_eval_pocket.slurm` (dev) |
| Pocket conditioning, oracle (upper bound) | the same script → `data/pockets_oracle.json`: heavy-chain residues within 6 Å of the crystal peptide (median 43) | the same job |
| Gradual MSA removal | `src/msa_schedule.py` (env `MSA_SCHEDULE`, `MSA_VAL_DEPTH`): the cap per epoch. The batch's MSA depth is checked against the cap and the run stops on a mismatch. | gpu-a100: `taper` (2048 → 512 → 128 → 32 → 8 → 1) and `abrupt` (1 from the start). Both use the batch-9, 3e-4 recipe and validate at depth 1 |
| Smoke test of all three | `scripts/ls6_round6_smoke.slurm` | dev, in the same job as the pocket eval |

The MSA arms validate at depth 1, so their reference is pretrained at depth 1
(`runs/eval_msa/msa1`), not `eval_val99/pretrained`.

Caveat for reading the pocket results: in training, Boltz marked about 4 pocket
tokens per conditioned sample (geometric sampling, `featurizer.py` ~474). Here we
mark 34 (pseudo-sequence) or about 43 (oracle). So the conditioning is outside the
training distribution, and a null result does not rule out a well-matched pocket.
