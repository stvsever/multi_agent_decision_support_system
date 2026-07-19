"""
Generate a self-contained interactive HTML viewer for an ontology.

The viewer renders the DOMAIN -> SUBDOMAIN -> FEATURE hierarchy with D3 and supports:
- expand / collapse on node click,
- drag to reposition nodes, plus zoom / pan,
- three layouts: top-down, left-right, and radial,
- a details panel (label, definition, path) and text search.

The ontology data is embedded inline, so the only external dependency is the D3
library (loaded from a CDN when the file is opened in a browser).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _to_d3(ontology: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": ontology.get("dataset", "ontology"),
        "kind": "root",
        "children": [
            {
                "name": d["label"], "id": d["id"], "kind": "domain",
                "definition": d.get("definition", ""),
                "children": [
                    {
                        "name": s["label"], "id": s["id"], "kind": "subdomain",
                        "definition": s.get("definition", ""),
                        "children": [
                            {"name": f["label"], "id": f["id"], "kind": "feature",
                             "definition": f.get("definition", ""), "units": f.get("units")}
                            for f in s["features"]
                        ],
                    }
                    for s in d["subdomains"]
                ],
            }
            for d in ontology["domains"]
        ],
    }


_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ - Ontology Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
  :root{--bg:#0b0f17;--panel:#131a26;--border:#243044;--txt:#e6edf6;--muted:#93a2b8;--pri:#6366f1;}
  *{box-sizing:border-box} html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
    font-family:Inter,system-ui,Segoe UI,sans-serif}
  header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--border);
    background:var(--panel);flex-wrap:wrap}
  header h1{font-size:15px;margin:0;font-weight:700;letter-spacing:.3px}
  header .sub{color:var(--muted);font-size:12px}
  .controls{display:flex;gap:8px;margin-left:auto;align-items:center;flex-wrap:wrap}
  button{background:#1b2434;color:var(--txt);border:1px solid var(--border);border-radius:7px;
    padding:7px 12px;font-size:12px;cursor:pointer;font-weight:600}
  button.active{background:var(--pri);border-color:var(--pri);color:#fff}
  button:hover{border-color:var(--pri)}
  input#search{background:#0d1420;border:1px solid var(--border);color:var(--txt);border-radius:7px;
    padding:7px 10px;font-size:12px;width:170px}
  #wrap{display:flex;height:calc(100vh - 55px)}
  #chart{flex:1;overflow:hidden}
  #side{width:290px;border-left:1px solid var(--border);background:var(--panel);padding:16px;overflow:auto}
  #side h2{font-size:13px;margin:0 0 6px;color:var(--pri);text-transform:uppercase;letter-spacing:.6px}
  #side .kind{display:inline-block;font-size:10px;padding:2px 8px;border-radius:999px;background:#1b2434;
    color:var(--muted);margin-bottom:8px}
  #side .name{font-size:16px;font-weight:700;margin-bottom:8px}
  #side .def{font-size:13px;color:var(--muted);line-height:1.5}
  #side .path{font-size:11px;color:var(--muted);margin-top:12px;word-break:break-word;font-family:ui-monospace,monospace}
  .link{fill:none;stroke:#33415a;stroke-width:1.3px;opacity:.7}
  .node circle{stroke-width:2px;cursor:pointer}
  .node text{font-size:11px;fill:var(--txt);paint-order:stroke;stroke:var(--bg);stroke-width:3px}
  .legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px;font-size:11px;color:var(--muted)}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
  .hint{font-size:11px;color:var(--muted);margin-top:10px;line-height:1.5}
</style></head>
<body>
<header>
  <h1>__TITLE__</h1><span class="sub">ontology explorer</span>
  <div class="controls">
    <input id="search" placeholder="search feature...">
    <button id="l-td" class="active" onclick="setLayout('td')">Top-down</button>
    <button id="l-lr" onclick="setLayout('lr')">Left-right</button>
    <button id="l-rad" onclick="setLayout('rad')">Radial</button>
    <button onclick="expandAll()">Expand all</button>
    <button onclick="collapseAll()">Collapse</button>
    <button onclick="resetView()">Reset</button>
  </div>
</header>
<div id="wrap">
  <div id="chart"></div>
  <div id="side">
    <h2>Details</h2>
    <div id="d-kind" class="kind">root</div>
    <div id="d-name" class="name">__TITLE__</div>
    <div id="d-def" class="def">Click any node to inspect it. Click a filled node to expand or collapse.</div>
    <div id="d-path" class="path"></div>
    <div class="legend" id="legend"></div>
    <div class="hint">Drag nodes to reposition. Scroll to zoom, drag background to pan.</div>
  </div>
</div>
<script>
const DATA = __DATA__;
const palette = d3.schemeCategory10;
let layout = 'td';
const chart = document.getElementById('chart');
let W = chart.clientWidth || 900, H = chart.clientHeight || 640;
const svg = d3.select('#chart').append('svg').attr('width', W).attr('height', H);
const g = svg.append('g');
const zoom = d3.zoom().scaleExtent([0.2, 3]).on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

const root = d3.hierarchy(DATA);
root.x0 = H/2; root.y0 = 0;
let domainColor = {};
root.children && root.children.forEach((d,i)=> domainColor[d.data.id||d.data.name] = palette[i%10]);
function colorOf(d){ let n=d; while(n.depth>1) n=n.parent; return n.depth===1 ? domainColor[n.data.id||n.data.name] : '#6366f1'; }
function collapse(d){ if(d.children){ d._children=d.children; d._children.forEach(collapse); d.children=null; } }
root.children && root.children.forEach(d => d.children && d.children.forEach(s => { /* keep 2 levels open */ }));
// start: domains + subdomains open, features collapsed
root.descendants().forEach(d=>{ if(d.depth>=2 && d.children){ d._children=d.children; d.children=null; }});

const legend = d3.select('#legend');
(root.children||[]).forEach(d=>{ legend.append('span').html(
  `<span class="dot" style="background:${domainColor[d.data.id||d.data.name]}"></span>${d.data.name}`); });

function setLayout(l){ layout=l; ['td','lr','rad'].forEach(k=>document.getElementById('l-'+k).classList.toggle('active',k===l)); update(root); resetView(); }

function tree(){
  if(layout==='rad') return d3.tree().size([2*Math.PI, Math.min(W,H)/2-90]).separation((a,b)=>(a.parent==b.parent?1:2)/a.depth);
  if(layout==='lr') return d3.tree().nodeSize([26, 200]);
  return d3.tree().nodeSize([150, 90]);
}
function project(d){
  if(layout==='rad'){ const a=d.x-Math.PI/2, r=d.y; return [Math.cos(a)*r, Math.sin(a)*r]; }
  if(layout==='lr') return [d.y, d.x];
  return [d.x, d.y];
}
const linkGen = ()=> layout==='rad'
  ? d3.linkRadial().angle(d=>d.x).radius(d=>d.y)
  : d3.linkHorizontal().x(d=>layout==='lr'?d.y:d.x).y(d=>layout==='lr'?d.x:d.y);

function update(source){
  const t = tree(); t(root);
  const nodes = root.descendants(), links = root.links();
  const dur = 400;

  const link = g.selectAll('path.link').data(links, d=>d.target.data.id||d.target.data.name);
  link.enter().append('path').attr('class','link').merge(link).transition().duration(dur)
    .attr('d', d=>{
      if(layout==='rad') return linkGen()(d);
      const s=project(d.source), t2=project(d.target);
      return `M${s[0]},${s[1]}C${(s[0]+t2[0])/2},${s[1]} ${(s[0]+t2[0])/2},${t2[1]} ${t2[0]},${t2[1]}`;
    });
  link.exit().remove();

  const node = g.selectAll('g.node').data(nodes, d=>d.data.id||d.data.name);
  const nEnter = node.enter().append('g').attr('class','node')
    .attr('transform', d=>{const p=project(source); return `translate(${p[0]},${p[1]})`;})
    .on('click',(e,d)=>{ select(d); if(d.children){d._children=d.children;d.children=null;} else if(d._children){d.children=d._children;d._children=null;} update(d); })
    .call(d3.drag().on('drag',function(e,d){ if(layout==='rad')return; d.x=(layout==='lr')?e.y:e.x; d.y=(layout==='lr')?e.x:e.y; d3.select(this).attr('transform',`translate(${project(d)[0]},${project(d)[1]})`); redrawLinks(); }));
  nEnter.append('circle').attr('r',0);
  nEnter.append('text').attr('dy','0.31em').text(d=>d.data.name)
    .attr('x',d=>d.children||d._children?-10:10).attr('text-anchor',d=>d.children||d._children?'end':'start');

  const nAll = nEnter.merge(node);
  nAll.transition().duration(dur).attr('transform', d=>{const p=project(d); return `translate(${p[0]},${p[1]})`;});
  nAll.select('circle').transition().duration(dur).attr('r', d=>d.depth===0?7:d.depth===1?9:d.depth===2?6:4)
    .attr('fill', d=> (d._children? colorOf(d): (d.depth===3?'#0d1420':colorOf(d))))
    .attr('stroke', d=>colorOf(d));
  nAll.select('text').attr('opacity', d=> d.depth<=2 || layout!=='td' ? 1 : 0.85)
    .attr('x',d=>d.children||d._children?-11:11).attr('text-anchor',d=>d.children||d._children?'end':'start');
  node.exit().transition().duration(dur).attr('transform',`translate(${project(source)[0]},${project(source)[1]})`).remove();

  nodes.forEach(d=>{d.x0=d.x;d.y0=d.y;});
}
function redrawLinks(){
  g.selectAll('path.link').attr('d', d=>{
    if(layout==='rad') return linkGen()(d);
    const s=project(d.source), t2=project(d.target);
    return `M${s[0]},${s[1]}C${(s[0]+t2[0])/2},${s[1]} ${(s[0]+t2[0])/2},${t2[1]} ${t2[0]},${t2[1]}`;});
}
function select(d){
  document.getElementById('d-kind').textContent = d.data.kind||'node';
  document.getElementById('d-name').textContent = d.data.name;
  document.getElementById('d-def').textContent = d.data.definition || (d.data.units?('units: '+d.data.units):'No definition.');
  document.getElementById('d-path').textContent = d.ancestors().reverse().map(a=>a.data.name).join('  >  ');
}
function expandAll(){ root.each(d=>{ if(d._children){d.children=d._children;d._children=null;} }); update(root); }
function collapseAll(){ root.descendants().forEach(d=>{ if(d.depth>=1 && d.children){d._children=d.children;d.children=null;} }); update(root); }
function resetView(){ const k=layout==='td'?0.75:0.7; svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(layout==='rad'?W/2:(layout==='lr'?110:W/2), layout==='rad'?H/2:(layout==='lr'?H/2:70)).scale(k)); }
document.getElementById('search').addEventListener('input', e=>{
  const q=e.target.value.toLowerCase();
  g.selectAll('g.node').select('circle').attr('stroke-width', d=> q && (d.data.name.toLowerCase().includes(q)) ? 4:2)
    .attr('fill', d=> q && d.data.name.toLowerCase().includes(q) ? '#f59e0b' : (d._children?colorOf(d):(d.depth===3?'#0d1420':colorOf(d))));
});
function remeasure(){ W=chart.clientWidth||W; H=chart.clientHeight||H; svg.attr('width',W).attr('height',H); resetView(); }
window.addEventListener('resize', remeasure);
update(root); resetView();
// Re-measure once the flex layout has settled (chart width can be 0 at parse time).
requestAnimationFrame(remeasure); setTimeout(remeasure, 120);
</script>
</body></html>
"""


def write_viewer(ontology: Dict[str, Any], path: Path, title: str = "") -> None:
    data = _to_d3(ontology)
    title = title or ontology.get("dataset", "Ontology")
    html = (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__DATA__", json.dumps(data)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
