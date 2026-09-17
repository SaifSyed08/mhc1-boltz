# Peptide templates: Boltz-1 has no template pathway

Ernest's second suggestion (2026-09-03) was:

> I'd rather consider templates as possible additional reliable source. Check out
> if you can find a way to use templates for peptides, it might be very helpful.

The idea is sound — a short peptide is exactly the case where a structural prior
should help more than an MSA. But it cannot be done in Boltz-1 as a data task,
and it is worth being precise about why before any work is planned around it.

## The finding

**Boltz-1 v1.0.0 contains no template support of any kind.**

```
$ grep -rli "template" boltz-src/src/boltz/
(no matches)

$ grep -rli "template" boltz-src/
docs/training.md          <- "a template configuration file", i.e. a YAML example
```

One match in the entire tree, and it is the word "template" used to describe an
example config file. There is:

* no template feature in `data/feature/featurizer.py`,
* no template field in `data/types.py`,
* no template module on `Boltz1` — its children are `input_embedder`,
  `msa_module`, `pairformer_module`, `structure_module`, `distogram_module` and
  the small init/recycle heads, and nothing else.

## The `template_id` field is vestigial

Our manifest carries a `template_id` on every chain, which makes it look as
though templates are wired up. They are not:

```
chain record fields:
  chain_id, chain_name, mol_type, cluster_id, msa_id, template_id,
  num_residues, valid

template_id == msa_id in 6168/6168 chains
```

It is a copy of `msa_id` in **every chain of every record**, it comes from the
upstream manifest schema, and no code in `src/boltz/` ever reads it.
`src/step6_write.py` preserves it only because the schema has the field.

So there is nothing to populate and nothing that would consume it.

## What "adding templates" would actually mean

AlphaFold-2/3-style templates are a model component, not an input file:

1. a featurizer that turns a template structure into pairwise distance/angle
   features aligned to the query sequence,
2. a template embedding stack that reduces those into the pair representation,
3. an injection point in the trunk, before or alongside the MSA module.

All three would have to be written. Worse for this project: **the pretrained
`boltz1_conf.ckpt` has no weights for any of it.** A freshly initialised template
stack feeding a frozen trunk would start by injecting noise into `z`, and the
amount of training needed to make it useful is far beyond what this project's
compute can reach — see `reports/CLOUD_GPU.md`, where 100 optimizer steps already
costs roughly a full week of free-tier quota.

## Boltz-2 does support templates

Boltz-2 added template conditioning explicitly: single or multimeric templates,
by upload or PDB ID, with tunable influence from a soft structural prior up to a
near-fixed scaffold. It also has a faster, more memory-efficient trunk, which
would independently help the problem in `reports/CLOUD_GPU.md`.

That makes this a scoping question for Ernest rather than a task to start:

* Did he mean Boltz-2? If so this is a much larger change of direction — a
  different model, different checkpoint, and the dataset would need re-processing
  against Boltz-2's format — but templates come for free and the memory picture
  improves.
* Or does he want template support implemented in Boltz-1? That is a research
  project in itself and does not fit the current compute budget.
* Or was the suggestion aimed at a later stage of the work?

## Recommendation

Ask before building. The question is cheap to ask and the wrong answer is
expensive: implementing a template stack for Boltz-1 would be weeks of work
producing a randomly-initialised module that this project cannot afford to train.

Nothing here contradicts the underlying intuition. Peptides are short, their MSAs
are weak, and a structural prior should help. It is the *vehicle* that is wrong,
not the idea.

## Sources

* [Boltz-2](https://boltz.bio/boltz2) — template conditioning and controllability
* [Boltz-2 preprint](https://www.biorxiv.org/content/10.1101/2025.06.14.659707.full.pdf)
* [Boltz-2 FAQ, Rowan](https://www.rowansci.com/blog/boltz2-faq) — template
  enforcement vs soft guidance
