"""Step 3: assign a biological role to every entity of every candidate entry.

Roles: MHC_I_HEAVY, MHC_I_FUSION, B2M, PEPTIDE, TCR_A/B/G/D, TCR_UNK, OTHER.

Design notes
------------
MHC-I heavy chains are identified primarily by *sequence homology*, not by
description text, per the brief's "do not rely exclusively on chain names".
The reference panel is bootstrapped from the data itself:

  seed  = entities whose description is unambiguously an MHC-I heavy chain,
          length 240-400, and which are NOT fusion constructs
  panel = the 5-mer pool of those seed sequences
  call  = k-mer containment of a candidate against the panel

Bootstrapping means every species present (chicken, duck, bat, pig, ...) is
covered without hand-curating reference sequences, and it correctly recovers
truncated alpha1-alpha2-only constructs (~175-185 aa) that a length filter
tuned on full ectodomains would miss.

Excluding fusion constructs from the seed is essential: peptide-linker-b2m-MHC
single-chain trimers would otherwise pull b2m and peptide k-mers into the panel,
after which b2m scores 1.00 against it and the classifier collapses. With a
clean panel the separation is wide (MHC-I heavy p05=0.86 vs b2m max=0.13,
TCR max=0.04), so the 0.25 threshold sits in an empty region.
"""
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from seqlib import kmer_containment, kmers, sw_align

D = pathlib.Path(__file__).resolve().parents[1] / "data"
meta = json.loads((D / "rcsb_meta.json").read_text())
tcr3d = {r["pdbid"].upper(): r for r in json.loads((D / "tcr3d_mhc1_raw.json").read_text())}

K = 5
MHC_CALL = 0.25          # >= this -> MHC-I heavy
MHC_AMBIG = 0.10         # in [AMBIG, CALL) -> flag for manual inspection

# Unambiguous MHC-I heavy descriptions, used only to seed the homology panel.
SEED_DESC = re.compile(
    r"((HLA|H\-?2)\s*CLASS\s*I\b|MHC\s*CLASS\s*I\b|CLASS\s*I\s*HISTOCOMPAT)", re.I)
# Fusion / single-chain-trimer constructs must stay out of the seed.
FUSION_DESC = re.compile(r"MICROGLOBULIN|\bB2M\b|LINKER|SINGLE[\s\-]?CHAIN|CHIMERIC|SCT", re.I)
CLASS_II = re.compile(r"CLASS\s*II|\bDR(A|B)\b|\bDQ(A|B)\b|\bDP(A|B)\b|\bH2\-?[AE]\b", re.I)
B2M_DESC = re.compile(r"BETA[\s\-_]*2[\s\-_]*MICROGLOBULIN|\bB2M\b|BETA[\s\-]?MICROGLOBULIN", re.I)
B2M_UNI = {"P61769", "P01887", "P01884", "Q30611", "P07151", "Q3T0T4", "P21756"}
TCR_DESC = re.compile(r"T[\s\-]?CELL\s+RECEPTOR|\bTCR\b|T[\s\-]?CELL\s+ANTIGEN\s+RECEPTOR", re.I)

MHC_LEN = (150, 400)     # covers a1a2-only constructs through full ectodomain
FUSION_LEN = 400         # a "heavy chain" longer than this is a fusion, not a chain
FUSION_MIN = 320         # ...as is a shorter one whose description names fused partners


def canon(pe):
    s = pe["entity_poly"].get("pdbx_seq_one_letter_code_can") or ""
    return s.replace("\n", "").strip().upper()


def tcr_kind(desc):
    d = desc.upper()
    for pat, role in (
        (r"\bALPHA\b|\bTCRA\b|\bTRA\b", "TCR_A"),
        (r"\bBETA\b|\bTCRB\b|\bTRB\b", "TCR_B"),
        (r"\bGAMMA\b|\bTCRG\b|\bTRG\b", "TCR_G"),
        (r"\bDELTA\b|\bTCRD\b|\bTRD\b", "TCR_D"),
    ):
        if re.search(pat, d):
            return role
    return "TCR_UNK"


# ---------------------------------------------------------------- entity table
entries = {}
for pid, e in meta.items():
    ents = []
    for pe in e.get("polymer_entities") or []:
        ci = pe["rcsb_polymer_entity_container_identifiers"]
        seq = canon(pe)
        ents.append({
            "entity_id": str(ci.get("entity_id")),
            "auth_asym_ids": ci.get("auth_asym_ids") or [],
            "label_asym_ids": ci.get("asym_ids") or [],
            "desc": (pe["rcsb_polymer_entity"].get("pdbx_description") or "").strip(),
            "seq": seq,
            "length": len(seq),
            "polymer_type": pe["entity_poly"].get("rcsb_entity_polymer_type"),
            "uniprot": sorted({a.get("reference_database_accession")
                               for a in (pe.get("rcsb_polymer_entity_align") or [])
                               if a.get("reference_database_accession")}),
            "organism": [o.get("ncbi_scientific_name")
                         for o in (pe.get("rcsb_entity_source_organism") or [])],
            "role": None, "role_evidence": None, "mhc_homology": None,
        })
    nonpoly = []
    for ne in e.get("nonpolymer_entities") or []:
        ci = ne.get("rcsb_nonpolymer_entity_container_identifiers") or {}
        cc = (ne.get("nonpolymer_comp") or {}).get("chem_comp") or {}
        nonpoly.append({"entity_id": str(ci.get("entity_id")), "comp_id": cc.get("id"),
                        "name": cc.get("name"),
                        "auth_asym_ids": ci.get("auth_asym_ids") or [],
                        "label_asym_ids": ci.get("asym_ids") or []})
    entries[pid] = {"pdb_id": pid, "entities": ents, "nonpolymer": nonpoly, "flags": []}


# ------------------------------------------------- stage A: b2m / peptide / TCR
def stage_a(pid, ents):
    pep_ref = (tcr3d.get(pid, {}).get("peptide") or "").strip().upper()
    if pep_ref in ("NONE", "---"):
        pep_ref = ""
    for en in ents:
        d, s, L = en["desc"], en["seq"], en["length"]
        if en["polymer_type"] not in ("Protein", None):
            en["role"] = "OTHER"
            en["role_evidence"] = "polymer_type=%s" % en["polymer_type"]
            continue
        if (B2M_DESC.search(d) or (set(en["uniprot"]) & B2M_UNI)) and 80 <= L <= 130:
            en["role"], en["role_evidence"] = "B2M", "desc/uniprot + length"
            continue
        if pep_ref and L <= 40:
            if s == pep_ref:
                en["role"], en["role_evidence"] = "PEPTIDE", "exact TCR3D peptide"
                continue
            if pep_ref in s or (len(pep_ref) >= 4 and s in pep_ref):
                en["role"], en["role_evidence"] = "PEPTIDE", "substring match to TCR3D peptide"
                continue
            _, idt, alen = sw_align(s, pep_ref)
            if alen >= max(5, int(0.7 * len(pep_ref))) and idt >= 0.7:
                en["role"] = "PEPTIDE"
                en["role_evidence"] = "SW to TCR3D peptide id=%.2f alen=%d" % (idt, alen)
                continue
        if TCR_DESC.search(d) and 90 <= L <= 320:
            en["role"], en["role_evidence"] = tcr_kind(d), "desc"
            continue


for pid, rec in entries.items():
    stage_a(pid, rec["entities"])

# ------------------------------------------------------ stage B: homology panel
seed = [e for r in entries.values() for e in r["entities"]
        if e["role"] is None and 240 <= e["length"] <= 400
        and SEED_DESC.search(e["desc"]) and not FUSION_DESC.search(e["desc"])
        and not CLASS_II.search(e["desc"])]
panel = set()
for e in seed:
    panel |= kmers(e["seq"], K)
print("homology seed: %d entities -> panel of %d %d-mers" % (len(seed), len(panel), K))

# ------------------------------------- stage C: call MHC-I heavy by containment
amb = []
for pid, r in entries.items():
    for en in r["entities"]:
        if en["role"] is not None or en["polymer_type"] != "Protein":
            continue
        if CLASS_II.search(en["desc"]):
            continue
        sc = kmer_containment(en["seq"], panel, K)
        en["mhc_homology"] = round(sc, 4)
        is_fusion = en["length"] > FUSION_LEN or (
            en["length"] > FUSION_MIN and FUSION_DESC.search(en["desc"]))
        if sc >= MHC_CALL and is_fusion:
            en["role"] = "MHC_I_FUSION"
            en["role_evidence"] = ("kmer_containment=%.3f, length=%d, fused partners in "
                                   "description -> single-chain construct" % (sc, en["length"]))
            r["flags"].append("single_chain_fusion")
        elif sc >= MHC_CALL and MHC_LEN[0] <= en["length"] <= MHC_LEN[1]:
            en["role"] = "MHC_I_HEAVY"
            en["role_evidence"] = "kmer_containment=%.3f" % sc
        elif MHC_AMBIG <= sc < MHC_CALL:
            en["role"] = "OTHER"
            en["role_evidence"] = "ambiguous MHC homology=%.3f" % sc
            r["flags"].append("ambiguous_mhc_homology")
            amb.append((sc, pid, en["length"], en["desc"][:50]))

# ------------------------------- stage D: b2m carrying expression tags/linkers
# Constructs such as 5KNM/5IUE prepend a signal peptide, His8 tag and a 3C
# protease site to b2m, giving a 182 aa chain that stage A's 80-130 window
# rejects. Accept a longer chain when the description/UniProt says b2m AND it is
# not MHC-homologous (which would mean a single-chain fusion, not a b2m subunit).
tagged = 0
for pid, r in entries.items():
    for en in r["entities"]:
        if en["role"] not in (None, "OTHER") or en["polymer_type"] != "Protein":
            continue
        if not (B2M_DESC.search(en["desc"]) or (set(en["uniprot"]) & B2M_UNI)):
            continue
        if not (80 <= en["length"] <= 260):
            continue
        sc = en["mhc_homology"]
        if sc is None:
            sc = kmer_containment(en["seq"], panel, K)
            en["mhc_homology"] = round(sc, 4)
        if sc < MHC_CALL:
            en["role"] = "B2M"
            en["role_evidence"] = ("desc/uniprot + length %d (tagged construct), "
                                   "MHC homology %.3f" % (en["length"], sc))
            r["flags"].append("b2m_with_expression_tag")
            tagged += 1
print("stage D recovered %d tagged b2m chains" % tagged)

for r in entries.values():
    for en in r["entities"]:
        if en["role"] is None:
            en["role"] = "OTHER"
            en["role_evidence"] = ("no rule matched"
                                   + ("; homology=%.3f" % en["mhc_homology"]
                                      if en["mhc_homology"] is not None else ""))

# ------------------------------------------------------------------- reporting
roles = collections.Counter(e["role"] for r in entries.values() for e in r["entities"])
print("\nrole counts: %s" % dict(roles.most_common()))

have_mhc = {p for p, r in entries.items() if any(e["role"] == "MHC_I_HEAVY" for e in r["entities"])}
have_pep = {p for p, r in entries.items() if any(e["role"] == "PEPTIDE" for e in r["entities"])}
have_fus = {p for p, r in entries.items() if any(e["role"] == "MHC_I_FUSION" for e in r["entities"])}
print("entries with MHC-I heavy   : %d/%d" % (len(have_mhc), len(entries)))
print("entries with peptide chain : %d/%d" % (len(have_pep), len(entries)))
print("entries with both          : %d/%d" % (len(have_mhc & have_pep), len(entries)))
print("entries that are fusions   : %d" % len(have_fus))
print("ambiguous homology calls   : %d %s" % (len(amb), sorted(amb, reverse=True)[:5]))
print("still no MHC-I heavy       : %s" % sorted(set(entries) - have_mhc - have_fus))

(D / "annotated_entities.json").write_text(json.dumps(entries, indent=1))
print("\nwrote %s" % (D / "annotated_entities.json"))
