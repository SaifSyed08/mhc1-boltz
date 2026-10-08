"""Score the peptide the way MHC-Fine does, one structure at a time.

Why this exists
---------------
MHC-Fine (Glukhov et al., Biophys J 2024) fine-tuned AlphaFold-Multimer v2.2 on
pMHC-I crystal structures and reports one headline number, "MHC-peptide RMSD":
superpose the prediction on the true structure using the MHC chain, then take
the RMSD over the peptide's CA atoms. Median on their test set: 1.59 A for the
base model, 0.66 A fine-tuned. Boltz's validation never measures the peptide on
its own. `lddt_pp` pools every inter-chain contact in the complex, most of them
heavy chain-b2m, so a better peptide can hide inside a flat `lddt_pp`. And every
Boltz metric is summed over structures before it reaches disk, so no number can
be traced back to the structure that produced it -- which is why the RMSD
excursion in reports/ls6/RESULT_round4.md could be sized but not named.

What it writes
--------------
One JSON line per validated structure, per rank, to
<output>/val_peptide_rank{r}.jsonl (append; ranks never share a file, so no
collective is needed). Each line has the record id, the step, and per diffusion
sample:

  pep_rmsd       MHC-Fine's metric: superpose on the groove CA, RMSD of peptide CA
  pep_internal   peptide CA RMSD after superposing the peptide on itself (shape only)
  groove_rmsd    groove CA RMSD after superposing on the groove
  ca_rmsd        all resolved protein CA, superposed with a proper rotation
  ca_rmsd_mirror the same, letting the superposition reflect. ca_rmsd >> ca_rmsd_mirror
                 means the sample is a mirror image, which lDDT cannot see
                 (a reflection preserves every distance)
  boltz_truth_dev  CA RMSD between the "truth" Boltz scored this sample against
                 and the real one (below). ~0 normally; above ~5 A means Boltz's
                 own rmsd/lDDT for this sample measured a disassembled complex
                 (a harmless swap of a whole twin copy reads ~0.5-1.1 A)
  boltz_rmsd     Boltz's own per-sample RMSD (all atoms, symmetry-corrected)
  lddt_pp, lddt_intra   Boltz's own per-sample lDDT. The `_pairs` fields are 0/1
                 flags (does the structure have such pairs at all), not counts:
                 Boltz reduces the totals to (total > 0) unless cardinality_weighted

Every score here is against the batch's own ground truth (`coords`), not the
symmetry-corrected truth Boltz uses for its metrics. Boltz's chain-swap search
(get_chain_symmetries) offers every same-entity chain of the full structure as
a replacement, including copies the dataset masked out, and keeps whichever
truth gives the best lDDT -- which a far-away copy can win, since it deletes
that chain's interface from the lDDT. 7ndt (in subset30) has a second complete
pMHC copy 67-120 A away. CA atoms have no within-residue symmetry and the kept
chains are distinct entities, so the raw truth loses nothing for these scores.

`pep_rmsd_best` (min over samples) is an oracle pick: it is chosen with the
answer in hand. MHC-Fine's numbers are one model per target picked by pLDDT,
and there is no confidence head here, so compare their medians with the median
over all samples (or sample 0), never with best-of-N.

Superposition uses the CA atoms of 180 heavy-chain residues starting at the
first one resolved in the true structure (the alpha1/alpha2 groove; MHC-Fine's
inputs contained only those domains; a few records carry an unresolved signal
peptide or leader, so the window cannot start at token 0), recorded as
`groove_start`. Chains are identified by length: heavy chain = longest protein
chain, which must exceed 150 residues; peptide = shortest, at most 30. A
structure that does not fit, or a batch Boltz skipped, is written with null
values and a `reason` -- never dropped silently.

Needs the local patch in model.py that stashes each batch's predictions on
`pl_module._val_stash`. Telemetry must never break a run: any failure becomes
a `reason` on the row, or a printed warning.
"""
import json
import os
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

try:
    from boltz.data import const
    PROTEIN = const.chain_type_ids["PROTEIN"]
except Exception:  # importable for tests without boltz
    PROTEIN = 0

GROOVE_RESIDUES = 180
MIN_HEAVY = 150
MAX_PEPTIDE = 30


def kabsch_rmsd(P, T, allow_reflection=False, Q=None, U=None):
    """Superpose P onto T (both [n, 3]) and return the RMSD of P->T, or, if Q
    and U are given, of Q->U after applying the same transform (score a part of
    the structure after superposing on another part). Proper rotation unless
    allow_reflection."""
    P, T = P.double(), T.double()
    pc, tc = P.mean(0), T.mean(0)
    H = (P - pc).T @ (T - tc)
    Uh, S, Vt = torch.linalg.svd(H)
    d = torch.ones(3, dtype=P.dtype, device=P.device)
    if not allow_reflection and torch.det(Vt.T @ Uh.T) < 0:
        d[2] = -1.0
    R = Vt.T @ torch.diag(d) @ Uh.T          # maps (x - pc) onto (y - tc)
    if Q is None:
        Q, U = P, T
    Q, U = Q.double(), U.double()
    moved = (Q - pc) @ R.T + tc
    return float(((moved - U) ** 2).sum(-1).mean().sqrt())


def record_ids(trainer):
    """Record ids in the order this rank validates them, or None.

    Lightning puts a DistributedSampler (shuffle=False) on the validation loader
    under DDP, so rank r sees dataset indices r, r+W, r+2W, ...; iterating the
    sampler gives exactly that list. Boltz's ValidationDataset maps a flat index
    onto its datasets the same way this walks them (training.py, __getitem__).
    """
    try:
        dl = trainer.val_dataloaders
        if isinstance(dl, (list, tuple)):
            dl = dl[0]
        ds = dl.dataset
        out = []
        for idx in iter(dl.sampler):
            for d in ds.datasets:
                n = len(d.manifest.records)
                if getattr(ds, "overfit", None) is not None:
                    n = min(n, ds.overfit)
                if idx < n:
                    out.append(str(d.manifest.records[idx].id))
                    break
                idx -= n
            else:
                out.append(None)
        return out
    except Exception as exc:
        print(f"[peptide_rmsd] record ids unavailable: {type(exc).__name__}: {exc}")
        return None


def _floats(x, n):
    if x is None:
        return [None] * n
    if torch.is_tensor(x):
        return [round(float(v), 4) for v in x.detach().flatten().tolist()]
    return [round(float(v), 4) for v in x]


def _median(v):
    v = sorted(v)
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def score_structure(batch, stash, b):
    """Per-sample scores for structure b of the batch. Returns a dict."""
    n = stash["n_samples"]
    asym = batch["asym_id"][b]
    mol = batch["mol_type"][b]
    pad = batch["token_pad_mask"][b].bool()
    frames = batch["frames_idx"][b]
    frame_ok = batch["frame_resolved_mask"][b].bool()
    true = batch["coords"][b, 0]                      # the real complex, never chain-swapped
    tmask = batch["atom_resolved_mask"][b].bool()

    prot = pad & (mol == PROTEIN)
    chains = {}
    for a in torch.unique(asym[prot]).tolist():
        chains[a] = torch.nonzero(prot & (asym == a)).flatten()   # token order
    if not chains:
        return {"reason": "no protein chains"}
    by_len = sorted(chains, key=lambda a: len(chains[a]))
    pep_a, hc_a = by_len[0], by_len[-1]
    out = {"hc_len": len(chains[hc_a]), "pep_len": len(chains[pep_a]),
           "n_protein_chains": len(chains)}
    if pep_a == hc_a or len(chains[hc_a]) <= MIN_HEAVY or len(chains[pep_a]) > MAX_PEPTIDE:
        out["reason"] = "chains do not look like heavy chain + peptide"
        return out

    def ca(tok):
        tok = tok[frame_ok[tok]]
        idx = frames[tok, 1]
        return idx[tmask[idx]]

    hc = chains[hc_a]
    hc_ok = torch.nonzero(frame_ok[hc] & tmask[frames[hc, 1]]).flatten()
    if len(hc_ok) == 0:
        out["reason"] = "no resolved heavy-chain CA"
        return out
    out["groove_start"] = int(hc_ok[0])
    groove_tok = hc[out["groove_start"]:out["groove_start"] + GROOVE_RESIDUES]
    g, p, a = ca(groove_tok), ca(chains[pep_a]), ca(torch.nonzero(prot).flatten())
    out["n_groove_ca"], out["n_pep_ca"], out["n_all_ca"] = len(g), len(p), len(a)
    if len(g) < 3 or len(p) < 3:
        out["reason"] = "too few resolved CA atoms"
        return out

    keys = ("pep_rmsd", "pep_internal", "groove_rmsd", "ca_rmsd", "ca_rmsd_mirror",
            "boltz_truth_dev")
    res = {k: [] for k in keys}
    for s in range(n):
        i = b * n + s
        pred = stash["pred"][i]
        res["pep_rmsd"].append(kabsch_rmsd(pred[g], true[g], Q=pred[p], U=true[p]))
        res["pep_internal"].append(kabsch_rmsd(pred[p], true[p]))
        res["groove_rmsd"].append(kabsch_rmsd(pred[g], true[g]))
        res["ca_rmsd"].append(kabsch_rmsd(pred[a], true[a]))
        res["ca_rmsd_mirror"].append(kabsch_rmsd(pred[a], true[a], allow_reflection=True))
        bt, bm = stash["true"][i], stash["true_mask"][i].bool()
        both = a[bm[a]]
        res["boltz_truth_dev"].append(kabsch_rmsd(bt[both], true[both]) if len(both) >= 3 else None)
    for k in keys:
        out[k] = [None if v is None else round(v, 4) for v in res[k]]
    out["pep_rmsd_best"] = min(out["pep_rmsd"])
    out["pep_rmsd_mean"] = round(sum(out["pep_rmsd"]) / n, 4)
    return out


class PeptideRMSD(Callback):
    def __init__(self, output_dir):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._ids = None
        self._best = []
        self._all = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._ids = record_ids(trainer)
        self._best = []
        self._all = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0):
        stash = getattr(pl_module, "_val_stash", None)
        pl_module._val_stash = None
        try:
            rank, ws = int(trainer.global_rank), int(trainer.world_size)
            bsz = int(batch["asym_id"].shape[0])
            for b in range(bsz):
                pos = batch_idx * bsz + b
                row = {
                    "record_id": (self._ids[pos] if self._ids and pos < len(self._ids) else None),
                    "global_step": int(trainer.global_step),
                    "epoch": int(trainer.current_epoch),
                    "sanity": bool(trainer.sanity_checking),
                    "rank": rank, "world_size": ws, "batch_idx": int(batch_idx),
                }
                if stash is None:
                    row["reason"] = "no predictions (batch skipped by Boltz)"
                else:
                    n = stash["n_samples"]
                    try:
                        row.update(score_structure(batch, stash, b))
                    except Exception as exc:
                        row["reason"] = f"scoring failed: {type(exc).__name__}: {exc}"
                    row["boltz_rmsd"] = _floats(stash.get("rmsds"), n)[b * n:(b + 1) * n]
                    for k, v in stash.get("lddt", {}).items():
                        row[f"lddt_{'pp' if k == 'protein_protein' else 'intra'}"] = \
                            _floats(v, n)[b * n:(b + 1) * n]
                    for k, v in stash.get("lddt_total", {}).items():
                        row[f"lddt_{'pp' if k == 'protein_protein' else 'intra'}_pairs"] = \
                            _floats(v, n)[b * n:(b + 1) * n]
                if "pep_rmsd_best" in row and not row["sanity"]:
                    self._best.append(row["pep_rmsd_best"])
                    self._all.extend(row["pep_rmsd"])
                with (self.dir / f"val_peptide_rank{rank}.jsonl").open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
        except Exception as exc:  # telemetry must never break a run
            print(f"[peptide_rmsd] row failed: {type(exc).__name__}: {exc}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._best and not trainer.sanity_checking:
            print(f"[peptide_rmsd] rank {trainer.global_rank} step {trainer.global_step}: "
                  f"{len(self._best)} structures, median MHC-peptide CA RMSD "
                  f"{_median(self._all):.2f} A over all {len(self._all)} samples "
                  f"(oracle best-of-N {_median(self._best):.2f} A; this rank's share, "
                  f"full set in val_peptide_rank*.jsonl)")
