"""Write predicted and true structures for chosen records, for PyMOL.

Why this exists
---------------
Validation keeps numbers, not coordinates, so a bad peptide RMSD could not be
looked at. Ernest (2026-10-12) asked for the unusual peptides to be inspected in
PyMOL. This callback writes, for every record listed in the env var
STRUCTURE_DUMP_IDS (comma-separated record ids, e.g. "7wj3,7t5m,7cir"):

  <output>/structures/<id>/truth.cif     the crystal, as the model was given it
                                         (masked chains removed; unresolved
                                         atoms left out)
  <output>/structures/<id>/sample<k>.cif each diffusion sample
  <output>/structures/<id>/view.pml      loads them, superposes every sample on
                                         the truth's alpha1/alpha2 groove CA (the
                                         MHC-Fine frame), colours the peptide

Each validation overwrites the files, so a training run leaves the last one.
Truth is batch["coords"][b, 0], the real complex, never Boltz's chain-swapped
truth (reports/UPSTREAM_PATCHES.md section 8).

Atom order: the tokenizer walks the unmasked chains, residues and atoms in
order, which is exactly Structure.remove_invalid_chains() order -- the same
assumption Boltz's own prediction writer makes (data/write/writer.py). A record
whose unpadded atom count does not match is skipped with a message, never
written wrong.
"""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from pytorch_lightning import Callback

from peptide_rmsd import GROOVE_RESIDUES, record_ids


def _target_dirs(trainer):
    """record id -> the dataset's target_dir, for every validation record."""
    out = {}
    try:
        dl = trainer.val_dataloaders
        if isinstance(dl, (list, tuple)):
            dl = dl[0]
        for d in dl.dataset.datasets:
            for r in d.manifest.records:
                out[str(r.id)] = Path(d.target_dir)
    except Exception as exc:
        print(f"[structure_dump] target dirs unavailable: {type(exc).__name__}: {exc}")
    return out


class StructureDump(Callback):
    def __init__(self, output_dir):
        self.dir = Path(output_dir) / "structures"
        self.want = {s.strip() for s in os.environ.get("STRUCTURE_DUMP_IDS", "").split(",") if s.strip()}
        self._ids = None
        self._dirs = {}
        if self.want:
            print(f"[structure_dump] will write {len(self.want)} record(s) to {self.dir}")

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.want:
            self._ids = record_ids(trainer)
            self._dirs = self._dirs or _target_dirs(trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0):
        # Runs BEFORE PeptideRMSD clears the stash only if registered first;
        # train.py registers this callback ahead of PeptideRMSD.
        if not self.want or trainer.sanity_checking:
            return
        stash = getattr(pl_module, "_val_stash", None)
        if stash is None:
            return
        try:
            bsz = int(batch["asym_id"].shape[0])
            for b in range(bsz):
                pos = batch_idx * bsz + b
                rid = self._ids[pos] if self._ids and pos < len(self._ids) else None
                if rid in self.want:
                    self._write(rid, batch, stash, b)
        except Exception as exc:  # telemetry must never break a run
            print(f"[structure_dump] failed: {type(exc).__name__}: {exc}")

    def _write(self, rid, batch, stash, b):
        from boltz.data.types import Interface, Structure
        from boltz.data.write.mmcif import to_mmcif

        pad = batch["atom_pad_mask"][b].bool()
        n = stash["n_samples"]
        true = batch["coords"][b, 0][pad].float().cpu().numpy()
        resolved = batch["atom_resolved_mask"][b][pad].bool().cpu().numpy()
        preds = [stash["pred"][b * n + k][pad].float().cpu().numpy() for k in range(n)]

        src = self._dirs.get(rid)
        if src is None:
            print(f"[structure_dump] {rid}: no target_dir"); return
        struct = Structure.load(src / "structures" / f"{rid}.npz").remove_invalid_chains()
        if len(struct.atoms) != len(true):
            print(f"[structure_dump] {rid}: {len(struct.atoms)} structure atoms vs "
                  f"{len(true)} unpadded batch atoms -- not written")
            return

        out = self.dir / rid
        out.mkdir(parents=True, exist_ok=True)

        def save(name, coords, present):
            atoms = struct.atoms.copy()
            atoms["coords"] = coords
            atoms["is_present"] = present
            residues = struct.residues.copy()
            residues["is_present"] = True
            st = replace(struct, atoms=atoms, residues=residues,
                         interfaces=np.array([], dtype=Interface))
            (out / f"{name}.cif").write_text(to_mmcif(st))

        save("truth", true, resolved)
        for k, p in enumerate(preds):
            save(f"sample{k}", p, np.ones(len(p), dtype=bool))
        (out / "view.pml").write_text(self._pml(struct, n, resolved))

    @staticmethod
    def _pml(struct, n, resolved):
        """Superpose on the groove exactly as peptide_rmsd does, colour the peptide.

        Groove = GROOVE_RESIDUES heavy-chain residues from the first one with a
        resolved CA (peptide_rmsd's groove_start), so PyMOL's frame is the
        metric's frame. Residues are numbered res_idx + 1 in the written mmCIF.
        """
        chains = list(struct.chains)
        hc = max(chains, key=lambda c: int(c["res_num"]))
        pep = min(chains, key=lambda c: int(c["res_num"]))
        res = struct.residues[hc["res_idx"]:hc["res_idx"] + hc["res_num"]]
        first = next((int(r["res_idx"]) for r in res if resolved[int(r["atom_center"])]), 0)
        hc, pep = str(hc["name"]), str(pep["name"])
        groove = f"chain {hc} and resi {first + 1}-{first + GROOVE_RESIDUES} and name CA"
        lines = ["load truth.cif, truth"]
        for k in range(n):
            lines += [f"load sample{k}.cif, sample{k}",
                      f"pair_fit sample{k} and {groove}, truth and {groove}"]
        lines += [
            "hide everything",
            f"show cartoon, chain {hc}",
            f"show sticks, chain {pep} and not hydro",
            "color grey80, truth",
            "color lightblue, sample*",
            f"color green, truth and chain {pep}",
            f"color magenta, sample* and chain {pep}",
            f"orient chain {pep}",
            f"# peptide = chain {pep} (green: crystal, magenta: samples); heavy chain = {hc}",
        ]
        return "\n".join(lines) + "\n"
