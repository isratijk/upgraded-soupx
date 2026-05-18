"""
generate_drawio.py
Outputs pipeline_diagram.drawio — open at diagrams.net, export PNG/PDF.
"""
import html as _html

W = 1400   # canvas width
PAD = 50   # left/right padding
BW = 1300  # usable box width

# ── Colors ─────────────────────────────────────────────────────────────────
C = {
    "data":  ("fill=#D6EAF8;strokeColor=#1F618D;", "#1F618D"),
    "share": ("fill=#EBF5FB;strokeColor=#1A5276;", "#1A5276"),
    "corr":  ("fill=#FDEBD0;strokeColor=#922B21;", "#922B21"),
    "metric":("fill=#FEF9E7;strokeColor=#7D6608;", "#7D6608"),
    "out":   ("fill=#D5F5E3;strokeColor=#1E8449;", "#1E8449"),
    "bl":    ("fill=#BDC3C7;strokeColor=#7F8C8D;", "#7F8C8D"),
    "au":    ("fill=#AED6F1;strokeColor=#1A5276;", "#1A5276"),
    "db":    ("fill=#A9DFBF;strokeColor=#1E8449;", "#1E8449"),
    "it":    ("fill=#FAD7A0;strokeColor=#935116;", "#935116"),
    "dx":    ("fill=#D7BDE2;strokeColor=#6C3483;", "#6C3483"),
    "gh":    ("fill=#A2D9CE;strokeColor=#0E6655;", "#0E6655"),
    "leg":   ("fill=#F8F9FA;strokeColor=#BDC3C7;", "#2C3E50"),
}

cells = []
nid = 2   # next ID (0 and 1 reserved)

def cell(value, x, y, w, h, style, vertex=True, parent="1",
         src=None, tgt=None, edge=False):
    global nid
    cid = str(nid); nid += 1
    # XML-escape the value so < > & " are safe inside an attribute
    ev = _html.escape(value, quote=True)
    if edge:
        cells.append(
            f'<mxCell id="{cid}" value="{ev}" style="{style}" '
            f'edge="1" source="{src}" target="{tgt}" parent="{parent}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    else:
        cells.append(
            f'<mxCell id="{cid}" value="{ev}" style="{style}" '
            f'vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
    return cid

def shared_box(label, sub, y, h, color_key):
    s, _ = C[color_key]
    base = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            "fontStyle=1;fontSize=13;verticalAlign=middle;")
    full_val = (f'<b>{label}</b><br/>'
                f'<span style="font-size:11px;font-weight:normal;font-style:italic;">'
                f'{sub}</span>')
    return cell(full_val, PAD, y, BW, h, base + s)

def dataset_box(label, sub, x, y):
    s, _ = C["data"]
    st = ("rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
          "fontStyle=1;fontSize=13;strokeWidth=2;")
    val = f'<b>{label}</b><br/><span style="font-size:11px;font-weight:normal;">{sub}</span>'
    return cell(val, x, y, 285, 75, st + s)

def pipe_box(label, src_file, y, h, color_key):
    s, sc = C[color_key]
    st = (f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
          f"fontStyle=1;fontSize=15;strokeWidth=2.5;")
    val = (f'<b>{label}</b><br/>'
           f'<span style="font-size:11px;font-weight:normal;font-style:italic;color:{sc};">'
           f'{src_file}</span>')
    return cell(val, 0, y, 185, h, st + s)  # x set per-call

def arrow(src, tgt, color="#5D6D7E", lw=2):
    st = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
          f"jettySize=auto;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
          f"entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
          f"strokeColor={color};strokeWidth={lw};endArrow=block;endFill=1;")
    return cell("", 0, 0, 0, 0, st, edge=True, src=src, tgt=tgt)

def harrow(src, tgt, color="#5D6D7E", lw=1.5):
    """Horizontal-first arrow for fanout."""
    st = (f"edgeStyle=elbowEdgeStyle;elbow=vertical;rounded=0;"
          f"strokeColor={color};strokeWidth={lw};endArrow=block;endFill=1;"
          f"exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
          f"entryX=0.5;entryY=0;entryDx=0;entryDy=0;")
    return cell("", 0, 0, 0, 0, st, edge=True, src=src, tgt=tgt)

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────
dy = 10
ds_xs = [PAD, PAD+345, PAD+690, PAD+1005]
d1 = dataset_box("Toy PBMC",      "62 cells · 226 genes",          ds_xs[0], dy)
d2 = dataset_box("PBMC 10k v3",   "11,769 cells · 33,538 genes",   ds_xs[1], dy)
d3 = dataset_box("HGMM Barnyard", "1,020 cells · 60,736 genes",    ds_xs[2], dy)
d4 = dataset_box("Fetal Liver",   "21,819 cells · 33,694 genes",   ds_xs[3], dy)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
load_y = 130
load = shared_box(
    "Data Loading  (io.py)",
    "Read raw count matrix (filtered) + empty droplets (raw)  ·  "
    "Build SoupChannel object  (soup_channel.py)",
    load_y, 80, "share")

for d in [d1, d2, d3, d4]:
    harrow(d, load, color="#1A5276", lw=1.8)

# ─────────────────────────────────────────────────────────────────────────────
# SOUP PROFILE
# ─────────────────────────────────────────────────────────────────────────────
soup_y = 255
soup = shared_box(
    "Soup Profile Estimation  (estimate_soup.py)",
    "Aggregate empty-droplet counts → normalise → per-gene ambient fraction  π_g  "
    "(fixed for all six pipelines)",
    soup_y, 80, "share")
arrow(load, soup, color="#1A5276", lw=2)

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
clust_y = 380
clust = shared_box(
    "Cluster Assignment  (set_properties.py)",
    "k-means on log-normalised counts → cell-to-cluster labels  ·  "
    "used by TF-IDF marker selection and EM topics (K)",
    clust_y, 80, "share")
arrow(soup, clust, color="#1A5276", lw=2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION HEADER
# ─────────────────────────────────────────────────────────────────────────────
hdr_st = ("text;html=1;strokeColor=none;fillColor=none;"
          "align=left;verticalAlign=middle;whiteSpace=wrap;"
          "fontStyle=1;fontSize=14;fontColor=#1A1A2E;")
cell("Contamination Estimation  —  6 Pipeline Variants",
     PAD, 478, 700, 30, hdr_st)

sep_st = ("line;strokeColor=#BDC3C7;fillColor=none;strokeWidth=1;"
          "entryX=1;entryY=0.5;entryDx=0;entryDy=0;")
cell("", PAD, 512, BW, 2, "shape=line;strokeColor=#BDC3C7;fillColor=none;strokeWidth=1;")

# ─────────────────────────────────────────────────────────────────────────────
# 6 PIPELINE BOXES
# ─────────────────────────────────────────────────────────────────────────────
pipe_y = 522
pipe_h = 155
pipe_xs = [PAD, PAD+218, PAD+436, PAD+654, PAD+872, PAD+1090]
pipe_cfg = [
    ("baseline",      "estimation.py",   "bl"),
    ("upg-auto",      "estimation.py",   "au"),
    ("upg-doublet",   "doublet.py",      "db"),
    ("upg-iterative", "iterative.py",    "it"),
    ("upg-decontx",   "decontx.py",      "dx"),
    ("upg-genehet",   "gene_het.py",     "gh"),
]

pipe_ids = []
for (lbl, src, ck), px in zip(pipe_cfg, pipe_xs):
    s, sc = C[ck]
    st = (f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
          f"fontStyle=1;fontSize=15;strokeWidth=2.5;")
    val = (f'<b>{lbl}</b><br/>'
           f'<span style="font-size:11px;font-weight:normal;font-style:italic;color:{sc};">'
           f'{src}</span>')
    pid = cell(val, px, pipe_y, 185, pipe_h, st + s)
    pipe_ids.append((pid, sc))
    # arrow from clust to each pipeline
    harrow(clust, pid, color=sc, lw=1.8)

# rho labels below each pipeline box
rho_texts = [
    ("global ρ̂", "#7F8C8D"),
    ("global ρ̂", "#1A5276"),
    ("global ρ̂<br/>(doublets: mean)", "#1E8449"),
    ("global ρ̂<br/>(2-round)", "#935116"),
    ("per-cell ρ̂ᵢ", "#6C3483"),
    ("per-cell ρ̂ᵢ", "#0E6655"),
]
for (pid, _), (rt, rc), px in zip(pipe_ids, rho_texts, pipe_xs):
    rho_st = (f"text;html=1;strokeColor=none;fillColor=none;"
              f"align=center;verticalAlign=middle;whiteSpace=wrap;"
              f"fontStyle=3;fontSize=11;fontColor={rc};")
    cell(rt, px, pipe_y + pipe_h + 5, 185, 40, rho_st)

# ─────────────────────────────────────────────────────────────────────────────
# COUNT CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
corr_y = 755
corr = shared_box(
    "Count Correction  (correction.py)",
    "Iterative weighted subtraction  ·  Per-cluster soup removal proportional to π_g  ·  "
    "Non-negative floor enforced",
    corr_y, 80, "corr")

for pid, sc in pipe_ids:
    harrow(pid, corr, color="#5D6D7E", lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
met_y = 885
met = shared_box(
    "Metrics Evaluation  (metrics.py  ·  downstream.py)",
    "M1 Cross-species fold  ·  M2 Marker fold-change  ·  M3 Cluster ARI  ·  "
    "M4 Batch entropy  ·  M5 HBB reduction  ·  GT-MAE  ·  GT-Pearson r  ·  "
    "EX Marker exclusivity  ·  M6 Silhouette  ·  M7 Spurious DE  ·  M8 Marker rank",
    met_y, 100, "metric")
arrow(corr, met, color="#5D6D7E", lw=2)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT BOXES
# ─────────────────────────────────────────────────────────────────────────────
out_y = 1040
out_cfg = [
    ("results.csv",       "All metrics per pipeline/dataset",          PAD + 35),
    ("Summary Figures",   "Plots 01–05: bar charts, heatmap, radar",   PAD + 435),
    ("Embedding Plots",   "Plots 06–07: UMAP & DE gene figures",       PAD + 835),
]
s_out, _ = C["out"]
out_st = (f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
          f"fontStyle=1;fontSize=13;strokeWidth=2;{s_out}")
out_ids = []
for lbl, sub, ox in out_cfg:
    val = (f'<b>{lbl}</b><br/>'
           f'<span style="font-size:11px;font-weight:normal;">{sub}</span>')
    oid = cell(val, ox, out_y, 365, 80, out_st)
    out_ids.append(oid)
    harrow(met, oid, color="#1E8449", lw=1.8)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
leg_y = 1175
leg_bg_st = ("rounded=1;arcSize=5;whiteSpace=wrap;html=1;"
             "fillColor=#F8F9FA;strokeColor=#BDC3C7;strokeWidth=1;")
cell("", PAD, leg_y, BW, 100, leg_bg_st)

# "Pipeline colour key:" label
hdr2_st = ("text;html=1;strokeColor=none;fillColor=none;"
           "align=left;verticalAlign=middle;fontStyle=1;fontSize=13;")
cell("Pipeline colour key:", PAD+10, leg_y+8, 230, 30, hdr2_st)

# Row 1: baseline, upg-auto, upg-doublet
row1_cfg = [
    ("baseline",     "bl", PAD+250),
    ("upg-auto",     "au", PAD+490),
    ("upg-doublet",  "db", PAD+730),
]
# Row 2: upg-iterative, upg-decontx, upg-genehet
row2_cfg = [
    ("upg-iterative","it", PAD+250),
    ("upg-decontx",  "dx", PAD+490),
    ("upg-genehet",  "gh", PAD+730),
]

for row_cfg, row_y in [(row1_cfg, leg_y+10), (row2_cfg, leg_y+55)]:
    for nm, ck, lx in row_cfg:
        fs, fc = C[ck]
        sw_st = (f"rounded=1;arcSize=20;strokeWidth=2;{fs}")
        cell("", lx, row_y, 26, 22, sw_st)
        txt_st = (f"text;html=1;strokeColor=none;fillColor=none;"
                  f"align=left;verticalAlign=middle;fontStyle=1;fontSize=12;"
                  f"fontColor={fc};")
        cell(nm, lx+32, row_y, 180, 22, txt_st)

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE XML
# ─────────────────────────────────────────────────────────────────────────────
body = "\n    ".join(cells)
xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2026-05-17" version="21.0.0">
  <diagram name="Pipeline" id="pipeline-diagram">
    <mxGraphModel dx="1422" dy="762" grid="0" gridSize="10"
                  guides="1" tooltips="1" connect="1" arrows="1"
                  fold="1" page="0" pageScale="1"
                  pageWidth="{W}" pageHeight="1300"
                  math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

out = "/Users/israt/Desktop/Upgraded-soupX_Final_Fahim_Before_clauded/plots/pipeline_diagram.drawio"
with open(out, "w", encoding="utf-8") as f:
    f.write(xml)
print(f"Saved: {out}")
