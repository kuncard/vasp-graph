"""
Generate 2D interactive VASP knowledge graph using vis.js.

Usage:
    python graph2d.py nodes.json edges.json --max 20 --output graph.html
"""
import json, argparse, os, re
from collections import defaultdict

COLORS = {
    "domain":        {"fill": "#5e81ac", "border": "#3b5d80", "label": "Domain (L1)"},
    "tutorial":      {"fill": "#7a9f6e", "border": "#4d6b42", "label": "Tutorial (L2)"},
    "parameter":     {"fill": "#c8963e", "border": "#8a5f28", "label": "Parameter"},
    "best_practice": {"fill": "#d08770", "border": "#9e5640", "label": "Heuristic (L3)"},
    "pitfall":       {"fill": "#bf616a", "border": "#8a3838", "label": "Constraint (L4)"},
    "generic":       {"fill": "#999999", "border": "#666666", "label": "Other"},
}
SUBTYPE_LABEL = {k: v["label"] for k, v in COLORS.items()}

def generate(nodes_path, edges_path, output_path, max_nodes=0):
    with open(nodes_path, encoding="utf-8") as f:
        nodes_list = json.load(f)
    with open(edges_path, encoding="utf-8") as f:
        edges_list = json.load(f)

    # Link contexts
    ctx_path = nodes_path.replace(".json", "_link_contexts.json")
    link_contexts = {}
    if os.path.exists(ctx_path):
        with open(ctx_path, encoding="utf-8") as f:
            link_contexts = json.load(f)

    nodes_map = {n["id"]: n for n in nodes_list}

    # Outgoing links
    out_links = defaultdict(list)
    for e in edges_list:
        if e["source"] in nodes_map and e["target"] in nodes_map:
            out_links[e["source"]].append(e["target"])

    # Select top nodes by degree, ensure struct-edge connections visible
    if max_nodes > 0 and len(nodes_list) > max_nodes:
        degree = defaultdict(int)
        for e in edges_list:
            if e["source"] in nodes_map and e["target"] in nodes_map:
                degree[e["source"]] += 1
                degree[e["target"]] += 1
        # Pick top N, then add their direct struct-edge neighbors (within limit)
        top_ids = set(sorted(degree, key=degree.get, reverse=True)[:max_nodes])
        # For nodes in top_ids without struct edges inside the set, add 1 neighbor
        for nid in list(top_ids):
            has_neighbor = False
            for e in edges_list:
                if e["relation"] == "wikilink":
                    continue
                if (e["source"] == nid and e["target"] in top_ids) or (e["target"] == nid and e["source"] in top_ids):
                    has_neighbor = True
                    break
            if not has_neighbor:
                # Add best neighbor (highest degree)
                best = None
                best_deg = 0
                for e in edges_list:
                    if e["relation"] == "wikilink":
                        continue
                    nb = e["target"] if e["source"] == nid else (e["source"] if e["target"] == nid else None)
                    if nb and nb not in top_ids and nb in degree:
                        if degree[nb] > best_deg:
                            best_deg = degree[nb]
                            best = nb
                if best and len(top_ids) < max_nodes + 3:
                    top_ids.add(best)
        nodes_list = [n for n in nodes_list if n["id"] in top_ids]
        nodes_map = {n["id"]: n for n in nodes_list}
        edges_list = [e for e in edges_list if e["source"] in nodes_map and e["target"] in nodes_map]
        print(f"Limited to {len(nodes_list)} nodes ({len(edges_list)} edges)")

    # Build vis nodes
    vis_nodes = []
    full_data = []
    for n in nodes_list:
        st = n.get("subtype", "generic")
        c = COLORS.get(st, COLORS["generic"])
        title = n.get("title", n["id"])
        vis_nodes.append({
            "id": n["id"], "label": title[:50],
            "color": {"background": c["fill"], "border": c["border"]},
        })

        qf = n.get("structured", {}).get("quick_facts")
        ts = n.get("structured", {}).get("tutorial_summary")
        link_ids = [t for t in out_links.get(n["id"], [])[:15] if t in nodes_map]
        links = []
        for t in link_ids:
            tgt = nodes_map[t]
            ctx = link_contexts.get(t, "")
            if ctx:
                ctx = ctx.replace("**", "").replace("$", "")
                ctx = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", ctx)
                ctx = re.sub(r"\[\[([^\]]+)\]\]", r"\1", ctx)
                if len(ctx) > 80:
                    ctx = ctx[:80].rsplit(" ", 1)[0] + "..."
            links.append({"title": tgt.get("title", t), "ctx": ctx})

        sf = n.get("structured", {})
        full_data.append({
            "id": n["id"], "title": title, "st": st,
            "content": (n.get("content", "") or "")[:2000],
            "tags": (n.get("tags", []) or [])[:10],
            "qf": {"type": qf["type"], "default": qf["default"],
                   "desc": qf.get("raw_description", "")[:1500]} if qf else None,
            "definition": sf.get("definition", ""),
            "options": sf.get("options", []),
            "warnings": sf.get("warnings", []),
            "tutorial": ts[:2000] if ts else None,
            "links": links,
        })

    # Build edges (struct only + all)
    struct_edges, all_edges = [], []
    seen = set()
    for e in edges_list:
        rel = e.get("relation", "wikilink")
        key = (e["source"], e["target"], rel)
        if key in seen:
            continue
        seen.add(key)
        ec = {"wikilink": "#c8ccd4", "belongs_to": "#5e81ac", "has_subcategory": "#b48ead",
              "contains": "#7a9f6e"}.get(rel, "#c8ccd4")
        edge = {"from": e["source"], "to": e["target"], "color": {"color": ec, "opacity": 0.5}}
        if rel != "wikilink":
            edge["arrows"] = "to"
        all_edges.append(edge)
        if rel != "wikilink":
            struct_edges.append(edge)

    # Load vis.js
    bundle = os.path.join(os.path.dirname(__file__) or ".", "vis_bundle.js")
    with open(bundle, encoding="utf-8") as f:
        vis_js = f.read()
    vis_js = vis_js.replace("</script>", "<\\/script>")
    vis_js = vis_js.replace("<script>", "<\\script>")

    # Stats
    st_counts = defaultdict(int)
    for n in nodes_list:
        st_counts[n.get("subtype", "generic")] += 1
    legend_parts = []
    for st in ["domain", "tutorial", "parameter", "best_practice", "pitfall", "generic"]:
        if st_counts.get(st):
            c = COLORS[st]
            legend_parts.append(
                f'<span><span class="dot" style="background:{c["fill"]};border-color:{c["border"]}"></span>'
                f'{c["label"]} ({st_counts[st]})</span>')
    legend = '<span class="sep">|</span>'.join(legend_parts)

    # ── HTML ──
    N = len(vis_nodes)
    SE = len(struct_edges)
    AE = len(all_edges)

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>VASP Knowledge Graph</title>
<style>
body{{margin:0;padding:0;background:#fafaf7;font-family:system-ui,sans-serif;overflow:hidden}}
#top{{z-index:10;position:absolute;top:0;left:0;right:0;background:#fffe;padding:6px 14px;border-bottom:1px solid #e8e6dd;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
#top h1{{font-size:14px;margin:0}}
#search{{padding:5px 10px;border:1px solid #ddd;border-radius:4px;width:180px;font-size:12px}}
#controls{{display:flex;gap:6px;align-items:center}}
#controls button{{font-size:10px;padding:3px 8px;border:1px solid #ddd;border-radius:4px;background:#fff;color:#666;cursor:pointer}}
#controls button.active{{background:#5e81ac;color:#fff;border-color:#5e81ac}}
.stats{{font-size:10px;color:#bbb}}
#legend{{font-size:10px;color:#888;display:flex;gap:10px;flex-wrap:wrap}}
#legend span{{display:flex;align-items:center;gap:3px}}
#legend .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid}}
#legend .sep{{color:#ccc;margin:0 2px}}
#container{{position:absolute;top:42px;bottom:0;left:0;right:0;background:#fafaf7}}
#detail{{position:absolute;top:52px;right:16px;z-index:20;background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;width:420px;max-height:80vh;overflow-y:auto;display:none;font-size:12px;line-height:1.6;box-shadow:0 2px 20px rgba(0,0,0,.08)}}
#detail h2{{font-size:15px;margin:0 0 2px}}
#detail .tp{{font-size:10px;color:#999;text-transform:uppercase;letter-spacing:.05em}}
#detail .tags{{margin:4px 0;display:flex;gap:3px;flex-wrap:wrap}}
#detail .tag{{font-size:9px;background:#eee;padding:1px 6px;border-radius:3px;color:#888}}
#detail .content{{margin-top:8px;color:#333;font-size:11px;line-height:1.5;max-height:500px;overflow-y:auto;background:#f8f8f6;padding:10px;border-radius:6px;white-space:pre-wrap;font-family:SFMono-Regular,Consolas,monospace}}
#detail .close{{position:absolute;top:8px;right:12px;font-size:18px;cursor:pointer;color:#bbb;background:none;border:none}}
</style>
<script>{vis_js}</script>
</head><body>
<div id="top">
  <h1>VASP Knowledge Graph</h1>
  <input id="search" placeholder="Search..." oninput="doSearch()"/>
  <span id="match-count" style="font-size:10px;color:#5e81ac;display:none"></span>
  <span id="controls">
    <button id="btn-struct" class="active" onclick="toggleEdges('struct')">Structure</button>
    <button id="btn-all" onclick="toggleEdges('all')">All Edges</button>
    <button id="btn-freeze" onclick="togglePhysics()">Freeze</button>
  </span>
  <span class="stats">{N} nodes · {SE} struct edges · {AE} total</span>
  <div id="legend">{legend}</div>
</div>
<div id="container"></div>
<div id="detail">
  <button class="close" onclick="document.getElementById('detail').style.display='none'">&times;</button>
  <h2 id="dt"></h2>
  <div class="tp" id="dty"></div>
  <div id="dtags" class="tags"></div>
  <div class="content" id="dc"></div>
</div>
<script>
var FULL = {json.dumps(full_data, ensure_ascii=False)};
var STRUCT_EDGES = {json.dumps(struct_edges, ensure_ascii=False)};
var ALL_EDGES = {json.dumps(all_edges, ensure_ascii=False)};
var ndMap = {{}}; FULL.forEach(function(n){{ ndMap[n.id] = n; }});

var nodes = new vis.DataSet({json.dumps(vis_nodes, ensure_ascii=False)});
var edges = new vis.DataSet(STRUCT_EDGES);
var showAllEdges = false, physicsOn = true;

var opts = {{
  nodes: {{shape:"dot",size:7,font:{{size:10,color:"#666"}},borderWidth:1.2}},
  edges: {{width:0.5,smooth:{{enabled:true,type:"continuous",roundness:0.4}},arrows:{{to:{{enabled:true,scaleFactor:0.4}}}}}},
  physics: {{stabilization:{{iterations:100,fit:true}},barnesHut:{{gravationalConstant:-20000,centralGravity:0.3,springLength:350,springConstant:0.005,damping:0.4}},maxVelocity:30,minVelocity:0.5}},
  interaction: {{hover:true,zoomView:true,dragView:true}}
}};
var network = new vis.Network(document.getElementById("container"), {{nodes:nodes,edges:edges}}, opts);

network.on("stabilizationIterationsDone", function() {{
  if(showAllEdges) return;  // Don't freeze when showing all edges
  network.setOptions({{physics:{{enabled:false}}}}); physicsOn=false;
}});

function toggleEdges(mode) {{
  showAllEdges = (mode==="all");
  edges.clear(); edges.add(mode==="all" ? ALL_EDGES : STRUCT_EDGES);
  document.getElementById("btn-struct").classList.toggle("active", mode==="struct");
  document.getElementById("btn-all").classList.toggle("active", mode==="all");
}}
function togglePhysics() {{
  physicsOn=!physicsOn; network.setOptions({{physics:{{enabled:physicsOn}}}});
  document.getElementById("btn-freeze").classList.toggle("active", !physicsOn);
}}

// Search with debounce
var searchTimer = null;
function doSearch() {{
  if(searchTimer) clearTimeout(searchTimer);
  searchTimer = setTimeout(filter, 300);
}}

function filter() {{
  var q = document.getElementById("search").value.toLowerCase().trim();
  var counter = document.getElementById("match-count");
  if(!q) {{
    counter.style.display = "none";
    var nu = [], eu = [];
    nodes.forEach(function(n){{ nu.push({{id:n.id,hidden:false,size:10,font:{{size:11}},borderWidth:1.5,color:{{border:null}}}}); }});
    edges.forEach(function(e){{ eu.push({{id:e.id,hidden:false}}); }});
    nodes.update(nu); edges.update(eu);
    network.fit({{animation:true}});
    return;
  }}
  // Find matching nodes (exact id/title > segment match > content match)
  var match = new Set();
  var exactMatch = new Set();
  var segMatch = new Set();
  var contentMatch = new Set();
  FULL.forEach(function(n) {{
    var idLow = n.id.toLowerCase();
    var titleLow = n.title.toLowerCase();
    if(idLow === q || titleLow === q) {{
      exactMatch.add(n.id);
    }} else if(idLow.split("_").indexOf(q)>=0 || titleLow.replace(/\\s+/g,"_").split("_").indexOf(q)>=0) {{
      segMatch.add(n.id);
    }} else if((idLow+" "+titleLow+" "+(n.content||"")+" "+(n.tags||[]).join(" ")).toLowerCase().indexOf(q)>=0) {{
      contentMatch.add(n.id);
    }}
  }});
  // Priority: exact > segment > content
  if(exactMatch.size > 0) {{
    match = exactMatch;
  }} else if(segMatch.size > 0) {{
    match = segMatch;
  }} else if(contentMatch.size <= 10) {{
    match = contentMatch;
  }}
  counter.style.display = "inline";
  counter.textContent = match.size + " match" + (match.size!==1?"es":"");

  // Build subgraph: matches + all 1-hop neighbors
  var visible = new Set(match);
  ALL_EDGES.forEach(function(e) {{
    if(match.has(e.from)) visible.add(e.to);
    if(match.has(e.to)) visible.add(e.from);
  }});

  var nodeUpdates = [];
  var edgeUpdates = [];
  nodes.forEach(function(n) {{
    var isMatch = match.has(n.id);
    if(isMatch) {{
      nodeUpdates.push({{id:n.id, hidden:false, size:25, font:{{size:15,color:"#000",bold:{{color:"#000"}}}}, borderWidth:3, color:{{border:"#000"}}}});
    }} else if(visible.has(n.id)) {{
      nodeUpdates.push({{id:n.id, hidden:false, size:10, font:{{size:11}}, borderWidth:1.5, color:{{border:null}}}});
    }} else {{
      nodeUpdates.push({{id:n.id, hidden:true, size:10, font:{{size:11}}}});
    }}
  }});
  edges.forEach(function(e) {{
    edgeUpdates.push({{id:e.id, hidden: !visible.has(e.from) || !visible.has(e.to)}});
  }});
  nodes.update(nodeUpdates);
  edges.update(edgeUpdates);
  if(visible.size>0) {{
    network.fit({{nodes:Array.from(visible),animation:true}});
    if(match.size <= 5) {{
      setTimeout(function(){{ network.fit({{nodes:Array.from(match),animation:true}}); }}, 700);
    }}
  }}
}}

// Double-click expand
network.on("doubleClick", function(p) {{
  if(p.nodes.length>0) {{
    var nid = p.nodes[0], connected = network.getConnectedNodes(nid);
    nodes.forEach(function(n) {{
      nodes.update({{id:n.id,hidden:connected.indexOf(n.id)<0&&n.id!==nid}});
    }});
    network.fit({{nodes:[nid].concat(connected),animation:true}});
  }}
}});

// Click detail
network.on("click", function(p) {{
  if(p.nodes.length>0) {{
    var n = ndMap[p.nodes[0]]; if(!n) return;
    var d = document.getElementById("detail"); d.style.display = "block";
    document.getElementById("dt").textContent = n.title||n.id;
    document.getElementById("dty").textContent = ({json.dumps(SUBTYPE_LABEL, ensure_ascii=False)})[n.st]||n.st;
    var td = document.getElementById("dtags"); td.innerHTML = "";
    (n.tags||[]).forEach(function(t){{ var s=document.createElement("span");s.className="tag";s.textContent=t;td.appendChild(s); }});

    // Build Markdown view
    var md = "";
    md += "---\\nid: "+n.id+"\\ntitle: \\""+(n.title||"")+"\\"\\ntype: "+n.st+"\\n";
    if(n.tags&&n.tags.length) md += "tags: ["+n.tags.join(", ")+"]\\n";
    md += "---\\n\\n";
    if(n.qf) {{
      md += "QUICK FACTS\\n\\n| | |\\n|---|---|\\n| Type | "+(n.qf.type||"?")+" |\\n| Default | "+(n.qf.default||"?")+" |\\n\\n"+(n.qf.desc||"");
      if(n.definition) md += "\\n\\nDefinition: "+n.definition;
      if(n.options&&n.options.length) {{
        md += "\\n\\nOptions\\n\\n";
        n.options.forEach(function(o){{ md += "- "+o.value; if(o.description) md += " -- "+o.description; md += "\\n"; }});
      }}
      if(n.warnings&&n.warnings.length) {{
        md += "\\n\\nWarnings\\n\\n";
        n.warnings.forEach(function(w){{ md += "! "+w+"\\n\\n"; }});
      }}
      md += "\\n\\nDescription\\n\\n"+(n.content||"");
    }} else if(n.tutorial) {{
      if(n.definition) md += "Definition: "+n.definition+"\\n\\n";
      md += "Summary\\n\\n"+n.tutorial;
      md += "\\n\\nContent\\n\\n"+(n.content||"");
    }} else {{
      if(n.definition) md += "Definition: "+n.definition+"\\n\\n---\\n\\n";
      md += (n.content||"(no content)");
    }}
    // Clean up formatting for plain text display
    md = md.replace(/^#### (.+)$/gm, "---- $1 ----");
    md = md.replace(/^### (.+)$/gm, "--- $1 ---");
    md = md.replace(/^## (.+)$/gm, "-- $1 --");
    md = md.replace(/\*\*(.+?)\*\*/g, "$1");
    md = md.replace(/\$([^$]+)\$/g, "$1");
    // Strip wikilink markup: [[page|title]] → title, [[page]] → page
    md = md.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2");
    md = md.replace(/\[\[([^\]]+)\]\]/g, "$1");
    if(n.links&&n.links.length) {{
      md += "\\n\\n## See Also\\n\\n";
      n.links.forEach(function(lk){{ md += "- "+lk.title; if(lk.ctx) md += " -- "+lk.ctx; md += "\\n"; }});
    }}
    document.getElementById("dc").textContent = md;
  }}
}});

network.on("click", function(p) {{ if(p.nodes.length===0&&p.edges.length===0) document.getElementById("detail").style.display="none"; }});
</script></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {output_path} ({len(html)/1024:.0f} KB)")
    print(f"  {N} nodes, {SE} struct edges + {AE-SE} wikilinks")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VASP 2D Knowledge Graph")
    p.add_argument("nodes"); p.add_argument("edges")
    p.add_argument("--output", "-o", default="VASP_Graph.html")
    p.add_argument("--max", "-n", type=int, default=0)
    args = p.parse_args()
    generate(args.nodes, args.edges, args.output, args.max)
