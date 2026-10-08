"""Synthetic check of src/peptide_rmsd.py: known answers by construction.

Plain script, no pytest: `python tests/test_peptide_rmsd.py`, exit 1 on any FAIL.
Needs torch and pytorch_lightning (the training venv), not boltz.
"""
import contextlib, io, json, pathlib, sys, tempfile, types

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from peptide_rmsd import PeptideRMSD, _median, kabsch_rmsd

torch.manual_seed(0)
lens = [200, 100, 9]                         # heavy chain, b2m, peptide
asym = torch.cat([torch.full((n,), i) for i, n in enumerate(lens)])
ntok = len(asym)
ca = torch.cumsum(torch.randn(ntok, 3) * 2.2, 0)               # random-walk CA trace
atoms = torch.stack([ca + torch.tensor([-1.2, 0.5, 0.0]), ca, ca + torch.tensor([1.3, 0.4, 0.1])], 1).reshape(-1, 3)
natom = atoms.shape[0] + 7                                       # padding atoms at the end
true = torch.zeros(natom, 3); true[:atoms.shape[0]] = atoms
frames = torch.stack([torch.arange(ntok) * 3, torch.arange(ntok) * 3 + 1, torch.arange(ntok) * 3 + 2], 1)
pep_atoms = slice(3 * 300, 3 * 309)

def rot(seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    return q * torch.sign(torch.det(q))                          # proper rotation

# Real truth: heavy-chain residues 0-4 unresolved (a leader/signal peptide, as
# in 7rtd), plus one unresolved CA inside the groove (residue 10).
mask = torch.zeros(natom, dtype=torch.bool); mask[:atoms.shape[0]] = True
for t in [0, 1, 2, 3, 4, 10]:
    mask[3 * t + 1] = False

s0 = true.clone(); s0[pep_atoms] += torch.tensor([1.0, 0.0, 0.0])   # peptide off by exactly 1 A
s0 = s0 @ rot(1).T + torch.tensor([5.0, -3.0, 2.0])
s1 = true.clone() * torch.tensor([-1.0, 1.0, 1.0])                 # mirror image
s1 = s1 @ rot(2).T
pred = torch.stack([s0, s1])

# What Boltz's symmetry correction may hand over as "truth": for sample 0 the
# peptide swapped for a masked copy 60 A away (7ndt's H1), for sample 1 the
# real complex, rigidly moved (Boltz aligns its truth to the prediction).
swapped = true.clone(); swapped[pep_atoms] += torch.tensor([60.0, 0.0, 0.0])
boltz_truth = torch.stack([swapped @ rot(4).T, true @ rot(5).T + 2.0])

batch = {"asym_id": asym[None], "mol_type": torch.zeros(1, ntok, dtype=torch.long),
         "token_pad_mask": torch.ones(1, ntok), "frames_idx": frames[None],
         "frame_resolved_mask": torch.ones(1, ntok, dtype=torch.bool),
         "coords": true[None, None], "atom_resolved_mask": mask[None].float()}
stash = {"pred": pred, "true": boltz_truth, "true_mask": torch.stack([mask, mask]), "n_samples": 2,
         "rmsds": [torch.tensor(0.5), torch.tensor(30.0)],
         "lddt": {"protein_protein": torch.tensor([0.9, 0.8])},
         "lddt_total": {"protein_protein": torch.tensor([1.0, 1.0])}}

rec = types.SimpleNamespace(id="1abc")
ds = types.SimpleNamespace(datasets=[types.SimpleNamespace(manifest=types.SimpleNamespace(records=[rec]))], overfit=None)
dl = types.SimpleNamespace(dataset=ds, sampler=[0])
trainer = types.SimpleNamespace(val_dataloaders=dl, global_rank=0, world_size=1, global_step=5,
                                current_epoch=0, sanity_checking=False)
module = types.SimpleNamespace(_val_stash=stash)

d = pathlib.Path(tempfile.mkdtemp())
cb = PeptideRMSD(d)
cb.on_validation_epoch_start(trainer, module)
cb.on_validation_batch_end(trainer, module, None, batch, 0)
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    cb.on_validation_epoch_end(trainer, module)
row = json.loads((d / "val_peptide_rank0.jsonl").read_text().splitlines()[0])
print(json.dumps(row))
print(printed.getvalue().strip())
ok = {
    "record id": row["record_id"] == "1abc",
    "no failure reason": "reason" not in row,
    "pep_rmsd sample0 == 1.0 despite swapped Boltz truth": abs(row["pep_rmsd"][0] - 1.0) < 1e-3,
    "groove_rmsd sample0 == 0": row["groove_rmsd"][0] < 1e-3,
    "pep_internal sample0 == 0": row["pep_internal"][0] < 1e-3,
    "mirror: ca_rmsd large": row["ca_rmsd"][1] > 5,
    "mirror: ca_rmsd_mirror ~0": row["ca_rmsd_mirror"][1] < 1e-3,
    "swap flagged: boltz_truth_dev sample0 large": row["boltz_truth_dev"][0] > 5,
    "no swap: boltz_truth_dev sample1 ~0": row["boltz_truth_dev"][1] < 1e-3,
    "best == min": row["pep_rmsd_best"] == min(row["pep_rmsd"]),
    "groove starts at first resolved CA": row["groove_start"] == 5,
    "unresolved groove CA dropped": row["n_groove_ca"] == 179,
    "boltz values passed": row["boltz_rmsd"] == [0.5, 30.0] and row["lddt_pp"] == [0.9, 0.8],
    "stash cleared": module._val_stash is None,
    "lengths": row["hc_len"] == 200 and row["pep_len"] == 9,
    "summary headlines all samples, labels oracle": "over all 2 samples" in printed.getvalue()
        and "oracle best-of-N" in printed.getvalue(),
    "median": _median([3, 1, 2]) == 2 and _median([4, 1, 3, 2]) == 2.5,
}
# a skipped batch: stash None -> row with reason, no crash
cb.on_validation_batch_end(trainer, module, None, batch, 0)
row2 = json.loads((d / "val_peptide_rank0.jsonl").read_text().splitlines()[1])
ok["skipped batch -> reason"] = "reason" in row2 and "pep_rmsd" not in row2
# kabsch sanity: identical up to rotation -> 0
P = torch.randn(20, 3); ok["kabsch rotation-invariant"] = kabsch_rmsd(P @ rot(3).T + 1.0, P) < 1e-6
for k, v in ok.items(): print(("PASS " if v else "FAIL ") + k)
sys.exit(0 if all(ok.values()) else 1)
