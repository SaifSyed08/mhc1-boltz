"""Merge validation state dumps from several runs into one correct result.

Usage:
    python src/combine_val_states.py runs/a/val_state.json runs/b/val_state.json

Why not just average the runs' reported metrics: Boltz updates each metric as
`MeanMetric.update(value, weight)`, where the weight is the number of atom pairs
contributing to that sample's lDDT. Samples are not equally weighted, so the mean
of two run-level means is only correct when both runs happen to carry identical
total weight. Summing the sufficient statistics and dividing once is exact
regardless.

Also prints the two numbers that actually matter for this project, and refuses to
print `val/lddt` without the warning attached -- see reports/BASELINE.md for why
that aggregate cannot exceed 0.385 on an all-protein dataset.
"""
import json
import pathlib
import sys


def main(paths):
    total = {}
    n_batches = 0
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        n_batches += d["n_batches_completed"]
        for name, st in d["metrics"].items():
            acc = total.setdefault(name, {"mean_value": 0.0, "weight": 0.0})
            acc["mean_value"] += st["mean_value"]
            acc["weight"] += st["weight"]
        print("  %-40s %d batches" % (pathlib.Path(p).name, d["n_batches_completed"]))

    merged = {}
    for name, st in sorted(total.items()):
        if st["weight"] > 0:
            merged[name] = st["mean_value"] / st["weight"]

    print("\nmerged over %d samples\n" % n_batches)
    headline = [
        ("lddt.intra_protein", "lDDT intra-protein (fold geometry)"),
        ("lddt.protein_protein", "lDDT protein-protein (peptide in groove)"),
        ("complex_lddt.intra_protein", "complex lDDT intra-protein"),
        ("complex_lddt.protein_protein", "complex lDDT protein-protein"),
        ("rmsd", "RMSD (A)"),
        ("best_rmsd", "best-of-N RMSD (A), N = diffusion_samples"),
    ]
    for key, label in headline:
        if key in merged:
            print("  %-45s %.4f" % (label, merged[key]))

    out = pathlib.Path("reports/val_merged.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"n_samples": n_batches, "metrics": merged}, indent=1, sort_keys=True))
    print("\nwrote %s" % out)
    print("NOTE: the aggregate `lddt` key is diluted by absent modality "
          "categories and cannot exceed 0.385 here. See reports/BASELINE.md.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
