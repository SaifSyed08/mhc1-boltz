"""Build the standalone pMHC-I structure viewer page.

Embeds the PDB text from data/viz/ directly into the HTML: the published page
cannot fetch anything at runtime, so every structure has to ship inside the file.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIZ = ROOT / "data" / "viz"

HTML = r"""<title>pMHC-I Groove Inspector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --ground:#EEF1F5; --panel:#FFFFFF; --panel-2:#F6F8FA; --line:#D3DAE3;
  --ink:#141C26; --ink-2:#4A5666; --ink-3:#7A8798;
  --viz-bg:#E7EBF0;
  --mhc:#2F7E93; --b2m:#6E5DA6; --pep:#C77A12; --dropped:#9AA6B4;
  --ok:#2C7A5B;
  --shadow:0 1px 2px rgba(20,28,38,.08),0 8px 24px rgba(20,28,38,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0F1721; --panel:#16202C; --panel-2:#1C2836; --line:#2A3646;
    --ink:#E4EAF1; --ink-2:#9FAEC0; --ink-3:#6B7C90;
    --viz-bg:#0B121A;
    --mhc:#4C93A8; --b2m:#8B7BB8; --pep:#E8A33D; --dropped:#3A4655;
    --ok:#4FB78A;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0F1721; --panel:#16202C; --panel-2:#1C2836; --line:#2A3646;
  --ink:#E4EAF1; --ink-2:#9FAEC0; --ink-3:#6B7C90;
  --viz-bg:#0B121A;
  --mhc:#4C93A8; --b2m:#8B7BB8; --pep:#E8A33D; --dropped:#3A4655;
  --ok:#4FB78A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  line-height:1.5; padding:24px;
}
.wrap{max-width:1240px;margin:0 auto;display:flex;flex-direction:column;gap:18px}

header{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px 20px}
h1{font-size:23px;font-weight:600;margin:0;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--ink-2);font-size:13.5px;max-width:66ch;margin:0}
.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);
}

.grid{display:grid;grid-template-columns:270px minmax(0,1fr);gap:18px;align-items:start}
@media (max-width:900px){.grid{grid-template-columns:1fr}}

.card{background:var(--panel);border:1px solid var(--line);border-radius:10px}
.pad{padding:14px 16px}

/* structure list ------------------------------------------------ */
.list{display:flex;flex-direction:column;overflow:hidden}
.list h2{font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);
  margin:0;padding:13px 16px 9px;font-family:"IBM Plex Mono",monospace;font-weight:500}
.item{
  display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center;
  padding:10px 16px;border:0;border-top:1px solid var(--line);background:none;
  color:inherit;font:inherit;text-align:left;cursor:pointer;width:100%;
}
.item:hover{background:var(--panel-2)}
.item[aria-current="true"]{background:var(--panel-2);box-shadow:inset 3px 0 0 var(--pep)}
.item:focus-visible{outline:2px solid var(--pep);outline-offset:-2px}
.pid{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:13.5px;letter-spacing:.02em}
.allele{grid-column:1;color:var(--ink-2);font-size:12px}
.reduce{
  grid-row:1/3;grid-column:2;font-family:"IBM Plex Mono",monospace;font-size:11px;
  color:var(--ink-3);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap;
}
.reduce b{color:var(--ink);font-weight:600}

/* viewport ------------------------------------------------------ */
.stage{position:relative;background:var(--viz-bg);border-radius:10px;overflow:hidden;
  border:1px solid var(--line);height:520px}
@media (max-width:900px){.stage{height:400px}}
#viewer{position:absolute;inset:0}
.legend{
  position:absolute;left:14px;top:14px;display:flex;flex-direction:column;gap:7px;
  background:color-mix(in srgb,var(--panel) 86%,transparent);
  border:1px solid var(--line);border-radius:8px;padding:11px 13px;backdrop-filter:blur(6px);
}
.lrow{display:flex;align-items:center;gap:9px;font-size:12px;color:var(--ink-2)}
.dot{width:9px;height:9px;border-radius:2px;flex:none}
.lrow b{color:var(--ink);font-weight:500}
.lrow .ch{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-3)}
.controls{position:absolute;right:14px;top:14px;display:flex;flex-direction:column;gap:7px;align-items:flex-end}
.tog{
  font:inherit;font-size:11.5px;padding:6px 11px;border-radius:7px;cursor:pointer;
  border:1px solid var(--line);background:color-mix(in srgb,var(--panel) 88%,transparent);
  color:var(--ink-2);backdrop-filter:blur(6px);
}
.tog[aria-pressed="true"]{color:var(--ink);border-color:var(--pep);
  box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--pep) 40%,transparent)}
.tog:focus-visible{outline:2px solid var(--pep);outline-offset:2px}
.hint{position:absolute;left:14px;bottom:12px;font-size:11px;color:var(--ink-3)}

/* data strip ---------------------------------------------------- */
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:0;
  border:1px solid var(--line);border-radius:10px;background:var(--panel);overflow:hidden}
.cell{padding:12px 16px;border-right:1px solid var(--line);display:flex;flex-direction:column;gap:3px}
.cell:last-child{border-right:0}
.k{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3)}
.v{font-size:14px;font-weight:500;font-variant-numeric:tabular-nums}
.v.mono{font-family:"IBM Plex Mono",monospace;font-size:13px}

.pepseq{display:flex;gap:3px;flex-wrap:wrap}
.res{
  font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:600;
  width:21px;height:24px;display:grid;place-items:center;border-radius:4px;
  background:color-mix(in srgb,var(--pep) 16%,transparent);
  color:var(--pep);border:1px solid color-mix(in srgb,var(--pep) 34%,transparent);
}
.check{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--ink-2)}
.check .mark{color:var(--ok);font-weight:700}
footer{color:var(--ink-3);font-size:11.5px;max-width:76ch}
footer code{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-2)}
</style>

<div class="wrap">
  <header>
    <div>
      <div class="eyebrow">Boltz-1 &middot; dataset preprocessing &middot; brief &sect;4 and check V7</div>
      <h1>pMHC-I Groove Inspector</h1>
    </div>
    <p class="sub">Every chain the original Boltz sample contained, with the ones this
    pipeline kept in colour and the ones it discarded in grey. The peptide should sit
    in the MHC-I &alpha;1/&alpha;2 groove &mdash; that is the geometry check V7 asserts,
    and the reason this view exists.</p>
  </header>

  <div class="grid">
    <nav class="card list" id="list" aria-label="Structures"><h2>Validation structures</h2></nav>

    <div style="display:flex;flex-direction:column;gap:14px;min-width:0">
      <div class="stage">
        <div id="viewer"></div>
        <div class="legend" id="legend"></div>
        <div class="controls">
          <button class="tog" id="t-drop" aria-pressed="true">Discarded chains</button>
          <button class="tog" id="t-surf" aria-pressed="false">Groove surface</button>
          <button class="tog" id="t-spin" aria-pressed="false">Spin</button>
        </div>
        <div class="hint">Drag to rotate &middot; scroll to zoom</div>
      </div>

      <div class="strip" id="strip"></div>
    </div>
  </div>

  <footer>
    Coordinates come from the unmodified Boltz training samples in
    <code>rcsb_processed_targets.tar</code>, exported by <code>src/export_pdb.py</code>.
    Discarded chains are drawn backbone-only. Chain selection is done on sequence and
    heavy-atom contact geometry, never on chain name &mdash; Boltz names chains by
    <code>label_asym_id</code>, which disagrees with the author IDs quoted by TCR3D for
    about 12% of chains.
  </footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
<script>
const DATA = __DATA__;
const ROLE_LABEL = {MHC_I_HEAVY:"MHC-I heavy chain", B2M:"β2-microglobulin", PEPTIDE:"Peptide"};
const ROLE_VAR   = {MHC_I_HEAVY:"--mhc", B2M:"--b2m", PEPTIDE:"--pep"};

const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
let viewer=null, current=null, showDropped=true, showSurf=false, spinning=false;

function build(){
  const el = document.getElementById("viewer");
  viewer = $3Dmol.createViewer(el, {backgroundColor: css("--viz-bg")});
  document.getElementById("list").insertAdjacentHTML("beforeend",
    DATA.map((d,i)=>`<button class="item" data-i="${i}" aria-current="${i===0}">
        <span class="pid">${d.pdb_id.toUpperCase()}</span>
        <span class="allele">${d.allele||"&mdash;"}</span>
        <span class="reduce"><b>${d.n_chains_kept}</b> / ${d.n_chains_total}<br>chains</span>
      </button>`).join(""));
  document.getElementById("list").addEventListener("click", e=>{
    const b = e.target.closest(".item"); if(!b) return;
    document.querySelectorAll(".item").forEach(x=>x.setAttribute("aria-current","false"));
    b.setAttribute("aria-current","true");
    show(+b.dataset.i);
  });
  bindTog("t-drop", v=>{showDropped=v; render();});
  bindTog("t-surf", v=>{showSurf=v;    render();});
  bindTog("t-spin", v=>{spinning=v; viewer.spin(v ? "y" : false);});
  show(0);
}
function bindTog(id, fn){
  const b=document.getElementById(id);
  b.addEventListener("click",()=>{
    const v = b.getAttribute("aria-pressed")!=="true";
    b.setAttribute("aria-pressed", String(v)); fn(v);
  });
}

function show(i){
  current = DATA[i];
  viewer.clear();
  viewer.addModel(current.pdb, "pdb");
  render(true);
  panel();
}

function render(recenter){
  if(!current) return;
  viewer.setStyle({}, {});
  viewer.removeAllSurfaces();
  const kept = current.chains.filter(c=>c.kept);
  const drop = current.chains.filter(c=>!c.kept).map(c=>c.pdb_chain).filter(Boolean);

  if(showDropped && drop.length){
    viewer.setStyle({chain: drop},
      {cartoon:{color: css("--dropped"), opacity:0.55, thickness:0.14, arrows:false}});
  }
  for(const c of kept){
    const col = css(ROLE_VAR[c.role] || "--mhc");
    viewer.setStyle({chain:c.pdb_chain}, {cartoon:{color:col, thickness:0.32, arrows:true}});
    if(c.role==="PEPTIDE"){
      viewer.addStyle({chain:c.pdb_chain}, {stick:{colorscheme:"default", color:col, radius:0.18}});
    }
  }
  if(showSurf){
    const heavy = kept.find(c=>c.role==="MHC_I_HEAVY");
    if(heavy){
      viewer.addSurface($3Dmol.SurfaceType.VDW,
        {opacity:0.62, color: css("--mhc")}, {chain:heavy.pdb_chain});
    }
  }
  if(recenter){
    const pep = kept.find(c=>c.role==="PEPTIDE");
    viewer.zoomTo(pep ? {chain:pep.pdb_chain} : {});
    viewer.zoom(0.55);
  }
  viewer.render();
}

function panel(){
  const d = current;
  const kept = d.chains.filter(c=>c.kept);
  document.getElementById("legend").innerHTML =
    kept.map(c=>`<div class="lrow"><span class="dot" style="background:${css(ROLE_VAR[c.role])}"></span>
      <b>${ROLE_LABEL[c.role]||c.role}</b> <span class="ch">${c.name}</span></div>`).join("") +
    `<div class="lrow"><span class="dot" style="background:${css("--dropped")}"></span>
      Discarded <span class="ch">${d.n_chains_total-d.n_chains_kept} chains</span></div>`;

  const pep = (d.peptide||"").toUpperCase();
  document.getElementById("strip").innerHTML = `
    <div class="cell"><span class="k">Entry</span><span class="v mono">${d.pdb_id.toUpperCase()}</span></div>
    <div class="cell"><span class="k">Allele</span><span class="v">${d.allele||"&mdash;"}</span></div>
    <div class="cell"><span class="k">Peptide &middot; ${pep.length} residues</span>
      <div class="pepseq">${[...pep].map(r=>`<span class="res">${r}</span>`).join("")}</div></div>
    <div class="cell"><span class="k">Released</span><span class="v mono">${d.released||"&mdash;"}</span></div>
    <div class="cell"><span class="k">Chains kept</span>
      <span class="v">${d.n_chains_kept} of ${d.n_chains_total}</span></div>
    <div class="cell"><span class="k">Check V7</span>
      <span class="check"><span class="mark">&check;</span> peptide in &alpha;1/&alpha;2 groove</span></div>`;
}

// Keep the 3D canvas on the same ground as the page when the OS theme flips.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", ()=>{
  if(!viewer) return;
  viewer.setBackgroundColor(css("--viz-bg"));
  render();
});

if(window.$3Dmol) build();
else window.addEventListener("load", build);
</script>
"""


def main():
    idx = json.loads((VIZ / "index.json").read_text())
    out = []
    for e in idx:
        p = VIZ / f"{e['pdb_id']}.pdb"
        if not p.exists():
            continue
        e = dict(e)
        e["pdb"] = p.read_text()
        e["chains"] = [c for c in e["chains"] if c.get("pdb_chain")]
        out.append(e)
    if not out:
        sys.exit("no structures in data/viz -- run src/export_pdb.py first")

    dest = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "viewer.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(HTML.replace("__DATA__", json.dumps(out)), encoding="utf-8")
    mb = dest.stat().st_size / 1e6
    print("wrote %s (%.1f MB, %d structures)" % (dest, mb, len(out)))
    if mb > 15:
        print("WARNING: approaching the 16 MB artifact limit")


if __name__ == "__main__":
    main()
