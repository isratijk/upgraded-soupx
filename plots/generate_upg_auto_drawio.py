"""
generate_upg_auto_drawio.py
Detailed draw.io diagram for the upg-auto pipeline in plain English.
Open at app.diagrams.net, edit visually, export PNG.
"""
import html as _html

cells = []
nid = 2

def cell(value, x, y, w, h, style, edge=False, src=None, tgt=None, parent="1"):
    global nid
    cid = str(nid); nid += 1
    ev = _html.escape(str(value), quote=True)
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

def arrow(src, tgt, color="#2C3E50", lw=2.5):
    st = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;"
          f"strokeColor={color};strokeWidth={lw};"
          f"endArrow=block;endFill=1;")
    return cell("", 0, 0, 0, 0, st, edge=True, src=src, tgt=tgt)

def side_arrow(src, tgt, color="#7D6608", lw=1.5):
    st = (f"edgeStyle=elbowEdgeStyle;elbow=vertical;rounded=1;"
          f"strokeColor={color};strokeWidth={lw};strokeDasharray=6 3;"
          f"endArrow=open;endFill=0;")
    return cell("", 0, 0, 0, 0, st, edge=True, src=src, tgt=tgt)

def S(fill, stroke, bold=True, fs=13, italic=False):
    f = 1 if bold else 0
    it = 4 if italic else 0
    return (f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
            f"fontStyle={f+it};fontSize={fs};verticalAlign=top;")

SINPUT  = S("#D6EAF8", "#1F618D")
SSTEP   = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;strokeWidth=2.5;fontStyle=1;fontSize=14;verticalAlign=top;"
SPARAM  = S("#FEF9E7", "#7D6608", bold=False, fs=11)
SNOTE   = S("#F4F6F7", "#717D7E", bold=False, fs=11)
SOUT    = S("#D5F5E3", "#1E8449")
SWARN   = S("#FDEBD0", "#935116", bold=False, fs=11)
SCOMP   = S("#FDFEFE", "#2C3E50", bold=False, fs=11)
STITLE  = ("text;html=1;strokeColor=none;fillColor=none;"
           "align=center;fontStyle=1;fontSize=18;fontColor=#1A1A2E;")
SSUB    = ("text;html=1;strokeColor=none;fillColor=none;"
           "align=center;fontStyle=2;fontSize=12;fontColor=#555577;")

SC = {
    1: ("#EAF2FF", "#2471A3"),
    2: ("#EAFAF1", "#1E8449"),
    3: ("#F5EEF8", "#6C3483"),
    4: ("#FEF5E7", "#935116"),
    5: ("#EBF5FB", "#1A5276"),
}

# ── TITLE ────────────────────────────────────────────────────────────────────
cell("Upgraded Automatic Contamination Estimation Pipeline",
     150, 12, 760, 36, STITLE)
cell("How upg-auto estimates the contamination fraction from the data",
     150, 50, 760, 24, SSUB)

# ── INPUT ────────────────────────────────────────────────────────────────────
inp = cell(
    "<b>INPUT: Single-Cell RNA Data</b><br/>"
    "<span style='font-weight:normal;font-size:11px;'>"
    "Gene count matrix for all cells<br/>"
    "Ambient soup gene profile computed from empty droplets<br/>"
    "Cell group assignments and total RNA count per cell"
    "</span>",
    200, 90, 660, 88, SINPUT)

# ── STEP 1 ───────────────────────────────────────────────────────────────────
y1 = 232
f1, s1 = SC[1]
step1 = cell(
    "<b>Step 1 - Group Cells by Cluster</b><br/>"
    "<span style='font-weight:normal;font-size:11px;'>"
    "Cells are grouped by their assigned cluster label<br/>"
    "Gene counts are summed across all cells within each cluster<br/>"
    "Total RNA counts are also summed per cluster<br/>"
    "This reduces the data from individual cells to cluster-level summaries,"
    " which makes the estimation faster and more robust"
    "</span>",
    200, y1, 660, 110, SSTEP + f"fillColor={f1};strokeColor={s1};")
arrow(inp, step1, color=s1)

# ── STEP 2 ───────────────────────────────────────────────────────────────────
y2 = 396
f2, s2 = SC[2]
step2 = cell(
    "<b>Step 2 - Select Diagnostic Marker Genes</b><br/>"
    "<span style='font-weight:normal;font-size:11px;'>"
    "For each gene, a specificity score is calculated across all clusters<br/>"
    "The score is high when a gene is common in one cluster but rare in all others<br/>"
    "Only genes above a specificity threshold of 1.0 are kept<br/>"
    "Among those, only genes that are also highly present in the ambient soup are kept<br/>"
    "At most 100 genes are selected to keep computation manageable"
    "</span>",
    200, y2, 660, 128, SSTEP + f"fillColor={f2};strokeColor={s2};")
arrow(step1, step2, color=s2)

p2 = cell(
    "<b>Settings used</b><br/>"
    "Specificity threshold: 1.0<br/>"
    "(original tool used 0.5,<br/>"
    "stricter here)<br/>"
    "Soup presence: top 10 percent<br/>"
    "Maximum genes: 100",
    20, y2, 165, 128, SPARAM)
side_arrow(step2, p2, color=s2)

# ── STEP 3 ───────────────────────────────────────────────────────────────────
y3 = 578
f3, s3 = SC[3]
step3 = cell(
    "<b>Step 3 - Identify Cell Groups That Do Not Express Each Gene</b><br/>"
    "<span style='font-weight:normal;font-size:11px;'>"
    "For each selected gene and each cluster, a statistical test checks whether"
    " the counts are higher than what ambient RNA alone would produce<br/>"
    "A false discovery correction is applied across all cells per gene<br/>"
    "If any cell in a cluster shows counts significantly above the ambient level,"
    " the entire cluster is excluded for that gene<br/>"
    "This leaves only clusters where all observed counts can be safely attributed"
    " to contamination rather than genuine expression"
    "</span>",
    200, y3, 660, 138, SSTEP + f"fillColor={f3};strokeColor={s3};")
arrow(step2, step3, color=s3)

p3 = cell(
    "<b>Settings used</b><br/>"
    "False discovery threshold:<br/>"
    "20 percent<br/>"
    "(original tool used 5 percent,<br/>"
    "relaxed here to allow<br/>"
    "more cluster-gene pairs<br/>"
    "into the estimation)",
    20, y3, 165, 138, SPARAM)
side_arrow(step3, p3, color=s3)

r3 = cell(
    "<b>Why exclude the whole cluster?</b><br/>"
    "If even one cell in a group<br/>"
    "genuinely expresses the gene,<br/>"
    "using that group would mix<br/>"
    "real expression with soup signal.<br/>"
    "This conservative rule prevents<br/>"
    "false contamination estimates.",
    875, y3, 185, 138, SWARN)
side_arrow(step3, r3, color="#935116")

# ── STEP 4 ───────────────────────────────────────────────────────────────────
y4 = 770
f4, s4 = SC[4]
step4 = cell(
    "<b>Step 4 - Estimate Contamination Rate for Each Gene and Cluster Pair</b><br/>"
    "<span style='font-weight:normal;font-size:11px;'>"
    "For each trusted gene and cluster combination from Step 3:<br/>"
    "Calculate how many counts of that gene the ambient soup alone would predict<br/>"
    "Divide the actual observed counts by the predicted ambient counts<br/>"
    "The result is an individual contamination rate for that gene-cluster pair<br/>"
    "Many of these individual estimates are then passed to Step 5 as evidence"
    "</span>",
    200, y4, 660, 120, SSTEP + f"fillColor={f4};strokeColor={s4};")
arrow(step3, step4, color=s4)

r4 = cell(
    "<b>Intuition:</b><br/>"
    "If the soup predicts 10 counts<br/>"
    "for a gene but we see 7, then<br/>"
    "70 percent of those counts came<br/>"
    "from ambient RNA in this cluster.<br/>"
    "Many such ratios are combined<br/>"
    "in Step 5 to get the final estimate.",
    875, y4, 185, 120, SNOTE)
side_arrow(step4, r4, color="#717D7E")

# ── STEP 5 ───────────────────────────────────────────────────────────────────
y5 = 944
f5, s5 = SC[5]
step5 = cell(
    "<b>Step 5 - Find the Best Global Contamination Fraction</b><br/>"
    "<span style='font-weight:normal;font-size:11px;'>"
    "A weak starting belief is set: contamination is probably around 5 percent"
    " with an uncertainty of 10 percent<br/>"
    "1001 candidate contamination values between 0 and 100 percent are tested<br/>"
    "For each candidate value, a total score is calculated by adding the starting"
    " belief score to the score from all the gene-cluster evidence pairs in Step 4<br/>"
    "The candidate value with the highest total score is selected as the final estimate<br/>"
    "Only values between 1 and 80 percent are considered as valid answers"
    "</span>",
    200, y5, 660, 160, SSTEP + f"fillColor={f5};strokeColor={s5};")
arrow(step4, step5, color=s5)

p5 = cell(
    "<b>Settings used</b><br/>"
    "Starting belief: 5 percent<br/>"
    "Uncertainty: 10 percent<br/>"
    "Number of values tested: 1001<br/>"
    "Step size: 0.1 percent<br/>"
    "Allowed range: 1 to 80 percent",
    20, y5, 165, 120, SPARAM)
side_arrow(step5, p5, color=s5)

r5 = cell(
    "<b>Why this approach is better:</b><br/>"
    "The original tool averaged<br/>"
    "the individual gene scores<br/>"
    "separately, which is statistically<br/>"
    "incorrect. This version combines<br/>"
    "all evidence into one single<br/>"
    "correct joint score before<br/>"
    "finding the best answer.",
    875, y5 + 10, 185, 140, SNOTE)
side_arrow(step5, r5, color="#1A5276")

# ── OUTPUT ───────────────────────────────────────────────────────────────────
y6 = 1160
out = cell(
    "<b>OUTPUT: Single Global Contamination Fraction for the Dataset</b><br/>"
    "<span style='font-weight:normal;font-size:11px;'>"
    "One contamination percentage is produced for the entire dataset<br/>"
    "This value is stored and passed to the correction step<br/>"
    "The correction step removes that proportion of ambient RNA from every cell"
    "</span>",
    200, y6, 660, 90, SOUT)
arrow(step5, out, color="#1E8449", lw=2.5)

# ── COMPARISON NOTE ──────────────────────────────────────────────────────────
cell(
    "<b>What changed compared to the original tool:</b><br/>"
    "The specificity filter for marker genes is stricter, selecting more reliable signals<br/>"
    "More cell groups are allowed into the estimation, giving the method more evidence to work with<br/>"
    "The scoring step correctly combines all evidence together instead of averaging each piece separately<br/>"
    "These three changes together produce 4 to 5 times higher and more accurate estimates on blood datasets",
    200, 1308, 660, 95, SCOMP)

# ── ASSEMBLE XML ─────────────────────────────────────────────────────────────
body = "\n    ".join(cells)
xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2026-05-17" version="21.0.0">
  <diagram name="upg-auto Detail" id="upg-auto-detail">
    <mxGraphModel dx="1200" dy="900" grid="0" gridSize="10"
                  guides="1" tooltips="1" connect="1" arrows="1"
                  fold="1" page="0" pageScale="1"
                  pageWidth="1100" pageHeight="1450"
                  math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

out_path = ("/Users/israt/Desktop/Upgraded-soupX_Final_Fahim_Before_clauded/"
            "plots/upg_auto_detail.drawio")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(xml)
print(f"Saved: {out_path}")
