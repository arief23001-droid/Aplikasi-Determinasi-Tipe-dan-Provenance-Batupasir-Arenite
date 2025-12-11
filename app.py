# app.py (FINAL - produce single PDF with combined interpretation + 4 combined plots)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import tempfile
import shutil
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import uuid

# ----------------------
# Config
# ----------------------
MAX_IMG_SIDE = 1200  # resize limit
st.set_page_config(layout="wide", page_title="Ternary Plotter — Final")
st.title("Ternary Plotter — multi diagram & 5-sample table")

# ----------------------
# Default files & vertices (original pixel-space)
# ----------------------
DEFAULT_IMG = {
    "Diagram 1 (Q-F-L)": "SEGITIGA_1.png",
    "Diagram 2 (Qm-F-L)": "SEGITIGA_2.png",
    "Diagram 3 (Q-F-L v2)": "SEGITIGA_3.png",
    "Diagram 4 (Qp-Lv-Ls)": "SEGITIGA_4.png",
}

DEFAULT_VERTS = {
    "Diagram 1 (Q-F-L)": {"Q": (5505, 2587), "F": (1572, 8790), "L": (9478, 8790)},
    "Diagram 2 (Qm-F-L)": {"Q": (5302, 2303), "F": (1390, 8567), "L": (9275, 8567)},
    "Diagram 3 (Q-F-L v2)": {"Q": (5444, 2323), "F": (1532, 8587), "L": (9437, 8567)},
    "Diagram 4 (Qp-Lv-Ls)": {"Qp": (5444, 2344), "Lv": (1532, 8587), "Ls": (9417, 8587)},
}

DEFAULT_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

# ----------------------
# Utility: safe image loader (PIL) -> returns (rgb_array, scale)
# ----------------------
def safe_load_image(path_or_file, max_side=MAX_IMG_SIDE):
    """Load image path or file-like, resize if too large. Return (rgb ndarray, scale)."""
    try:
        if hasattr(path_or_file, "read"):
            pil = Image.open(path_or_file)
        else:
            pil = Image.open(path_or_file)
    except Exception:
        # fallback blank white image
        return np.ones((600,600,3), dtype=np.uint8)*255, 1.0

    orig_w, orig_h = pil.size
    scale = 1.0
    max_orig = max(orig_w, orig_h)
    if max_orig > max_side:
        scale = float(max_side) / float(max_orig)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

    pil = pil.convert("RGBA")
    arr = np.array(pil)
    if arr.shape[-1] == 4:
        # composite alpha on white to avoid transparency issues
        alpha = arr[..., 3:4] / 255.0
        rgb = (arr[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    else:
        rgb = arr[..., :3].astype(np.uint8)
    return rgb, scale

# ----------------------
# Computation helpers
# ----------------------
def compute_qfl_from_row(row):
    Q = float(row["Qm"]) + float(row["Qp"])
    F = float(row["F"])
    L = float(row["Lv"]) + float(row["Ls"]) + float(row["Lm"])
    total = Q + F + L
    if total == 0:
        return (0.0, 0.0, 0.0)
    return (Q/total*100, F/total*100, L/total*100)

def compute_qm_fl_from_row(row):
    Qm = float(row["Qm"])
    F = float(row["F"])
    L = float(row["Lv"]) + float(row["Ls"]) + float(row["Lm"])
    total = Qm + F + L
    if total == 0:
        return (0.0, 0.0, 0.0)
    return (Qm/total*100, F/total*100, L/total*100)

def compute_qp_lv_ls_from_row(row):
    Qp = float(row["Qp"])
    Lv = float(row["Lv"])
    Ls = float(row["Ls"])
    total = Qp + Lv + Ls
    if total == 0:
        return (0.0, 0.0, 0.0)
    return (Qp/total*100, Lv/total*100, Ls/total*100)

def barycentric_to_pixel(weights, verts):
    # weights = (w1,w2,w3) in percent summing to 100
    w = np.array(weights) / 100.0
    v = np.array(verts)  # shape (3,2)
    px = float(w[0]*v[0][0] + w[1]*v[1][0] + w[2]*v[2][0])
    py = float(w[0]*v[0][1] + w[1]*v[1][1] + w[2]*v[2][1])
    return px, py

# ----------------------
# Interpretation logic (return label + prose explanation)
# ----------------------
def classify_qfl_arentite(q,f,l):
    if q >= 90: return "Quartzarenite"
    if f >= 25: return "Arkosic Arenite"
    if l >= 25: return "Lithic Arenite"
    if 5 <= f < 25 and q >= 50: return "Subarkose"
    if 5 <= l < 25 and q >= 50: return "Subarenite"
    return "Arenite (mixed/uncertain)"

ARENITE_EXPLANATION = {
    "Quartzarenite": "Menunjukkan sumber craton interior yang stabil dengan dominasi kuarsa akibat reworking.",
    "Arkosic Arenite": "Kontribusi feldspar tinggi, mengarah pada sumber granitik atau uplift kontinen.",
    "Lithic Arenite": "Kontribusi fragmen batuan tinggi — tipikal recycled orogenic/volkanik.",
    "Subarkose": "Campuran kuarsa dan feldspar; indikasi zona transisi atau uplift regional.",
    "Subarenite": "Kuarsa dominan dengan sedikit litik, kemungkinan daur ulang sedimen.",
    "Arenite (mixed/uncertain)": "Komposisi campuran — interpretasi tidak tegas."
}

def classify_qmfl_provenance(qm,f,l):
    if qm >= 70: return "Craton Interior / Stable Craton Source"
    if 40 <= qm < 70: return "Quartzose Recycled"
    if f >= 45: return "Dissected Arc / Magmatic Arc Source"
    if l >= 50: return "Lithic Recycled / Active Orogen"
    if 30 <= qm < 40 and f < 30: return "Transitional Continental"
    return "Mixed Provenance"

QMFL_EXPLANATION = {
    "Craton Interior / Stable Craton Source": "Sumber benua stabil (craton) dengan fragmen kuarsa mature.",
    "Quartzose Recycled": "Material dihasilkan dari daur ulang sedimen kuarsa (recycled orogenic).",
    "Dissected Arc / Magmatic Arc Source": "Kandungan feldspar tinggi menunjukkan pengaruh busur magmatik yang tererosi.",
    "Lithic Recycled / Active Orogen": "Kontribusi tinggi fragmen batuan, khas orogen aktif atau vulkanik.",
    "Transitional Continental": "Zona transisi antara platform benua dan area orogenik.",
    "Mixed Provenance": "Sumber campuran; tidak ada satu sumber dominan."
}

def classify_qfl_provenance_v2(q,f,l):
    if q >= 65 and f < 25: return "Craton Interior / Recycled"
    if f >= 40: return "Arc-derived (Dissected/Undissected Arc)"
    if l >= 45: return "Recycled Orogenic / Lithic-dominated"
    return "Mixed Provenance"

PROV3_EXPLANATION = {
    "Craton Interior / Recycled": "Pengaruh sumber benua mature yang direcycled.",
    "Arc-derived (Dissected/Undissected Arc)": "Kontribusi signifikan dari busur magmatik.",
    "Recycled Orogenic / Lithic-dominated": "Sumber dari fold-thrust belts/orogen yang tererosi.",
    "Mixed Provenance": "Sumber campuran."
}

def classify_qp_lv_ls(qp,lv,ls):
    if ls >= 55: return "Subduction Complex / Accretionary Prism"
    if lv >= 55: return "Volcanic Arc (juvenile arc source)"
    if qp >= 50: return "Stable Continental / Platform-dominated"
    if lv > ls and lv >= 30: return "Arc-influenced (volcanic-rich)"
    if ls > lv and ls >= 30: return "Lithic-dominated / Recycled Orogenic"
    return "Mixed Orogenic Source"

QP_EXPLANATION = {
    "Subduction Complex / Accretionary Prism": "Menunjukkan kontribusi material dari prisma akresi/subduksi (mélange, trench deposits).",
    "Volcanic Arc (juvenile arc source)": "Dominasi fragmen vulkanik; sumber busur magmatik.",
    "Stable Continental / Platform-dominated": "Dominasi kuarsa primer dari platform benua.",
    "Arc-influenced (volcanic-rich)": "Pengaruh busur magmatik tetapi tidak eksklusif.",
    "Lithic-dominated / Recycled Orogenic": "Kontribusi fragmen batuan kuat; recycled orogen.",
    "Mixed Orogenic Source": "Sumber campuran."
}

def interpret_sample(weights, diagram_key):
    # returns (label, explanation sentence)
    if diagram_key == "Diagram 1 (Q-F-L)":
        q,f,l = weights
        lbl = classify_qfl_arentite(q,f,l)
        expl = ARENITE_EXPLANATION.get(lbl, "")
        return lbl, expl
    if diagram_key == "Diagram 2 (Qm-F-L)":
        qm,f,l = weights
        lbl = classify_qmfl_provenance(qm,f,l)
        expl = QMFL_EXPLANATION.get(lbl, "")
        return lbl, expl
    if diagram_key == "Diagram 3 (Q-F-L v2)":
        q,f,l = weights
        lbl = classify_qfl_provenance_v2(q,f,l)
        expl = PROV3_EXPLANATION.get(lbl, "")
        return lbl, expl
    # diagram 4
    qp,lv,ls = weights
    lbl = classify_qp_lv_ls(qp,lv,ls)
    expl = QP_EXPLANATION.get(lbl, "")
    return lbl, expl

# ----------------------
# Sidebar + inputs
# ----------------------
st.sidebar.header("Diagram & image settings")
diagram_choice = st.sidebar.selectbox("Choose diagram to plot", list(DEFAULT_IMG.keys()))
uploaded = st.sidebar.file_uploader("Optional: upload an image to override selected diagram", type=["png","jpg","jpeg"])

# vertex labels and default numeric overrides
if diagram_choice == "Diagram 4 (Qp-Lv-Ls)":
    v1_label, v2_label, v3_label = "Qp","Lv","Ls"
else:
    v1_label, v2_label, v3_label = "Q","F","L"

vert_defaults = DEFAULT_VERTS[diagram_choice]
v1x = st.sidebar.number_input(f"{v1_label}_x", value=int(vert_defaults[v1_label][0]))
v1y = st.sidebar.number_input(f"{v1_label}_y", value=int(vert_defaults[v1_label][1]))
v2x = st.sidebar.number_input(f"{v2_label}_x", value=int(vert_defaults[v2_label][0]))
v2y = st.sidebar.number_input(f"{v2_label}_y", value=int(vert_defaults[v2_label][1]))
v3x = st.sidebar.number_input(f"{v3_label}_x", value=int(vert_defaults[v3_label][0]))
v3y = st.sidebar.number_input(f"{v3_label}_y", value=int(vert_defaults[v3_label][1]))

use_uploaded = uploaded is not None

# ----------------------
# Editable table (5 samples)
# ----------------------
st.markdown("### 1) Isi data mineral (5 sampel). Edit angka lalu tekan tombol PDF.")
default_table = pd.DataFrame({
    "Sample": [f"S{i+1}" for i in range(5)],
    "Qm": [30,25,20,40,15],
    "Qp": [5,10,5,2,8],
    "F":  [10,15,20,5,12],
    "Lv": [7,3,10,20,6],
    "Ls": [13,5,8,7,20],
    "Lm": [18,20,10,10,9],
})
st.info("Edit nilai-nilai pada tabel. Pastikan non-negative.")
edited = st.data_editor(default_table, num_rows="dynamic")
st.markdown("---")

# ----------------------
# Preload & resize diagram images into DIAGRAM_IMAGES: {key: (arr, scale)}
# ----------------------
DIAGRAM_IMAGES = {}
for key,path in DEFAULT_IMG.items():
    if use_uploaded and key == diagram_choice:
        arr, scale = safe_load_image(uploaded, MAX_IMG_SIDE)
    else:
        if os.path.exists(path):
            arr, scale = safe_load_image(path, MAX_IMG_SIDE)
        else:
            arr, scale = np.ones((600,600,3), dtype=np.uint8)*255, 1.0
    DIAGRAM_IMAGES[key] = (arr, scale)

def get_scaled_vertex_coords_for_diagram(diagram_key):
    arr, scale = DIAGRAM_IMAGES.get(diagram_key, (np.ones((600,600,3), dtype=np.uint8), 1.0))
    if diagram_key == "Diagram 4 (Qp-Lv-Ls)":
        vdef = DEFAULT_VERTS[diagram_key]
        return [(vdef["Qp"][0]*scale, vdef["Qp"][1]*scale),
                (vdef["Lv"][0]*scale, vdef["Lv"][1]*scale),
                (vdef["Ls"][0]*scale, vdef["Ls"][1]*scale)]
    # otherwise use user overrides v1x... (original pixel space) scaled to image
    return [(v1x*scale, v1y*scale), (v2x*scale, v2y*scale), (v3x*scale, v3y*scale)]

# ----------------------
# Create matplotlib figure for (diagram, row) - still used for single-streamlit preview
# ----------------------
def create_plot_figure_for_sample(diagram_key, row):
    img, scale = DIAGRAM_IMAGES.get(diagram_key, (np.ones((600,600,3), dtype=np.uint8),1.0))
    # choose which weight function to use
    if diagram_key == "Diagram 1 (Q-F-L)":
        w = compute_qfl_from_row(row)
    elif diagram_key == "Diagram 2 (Qm-F-L)":
        w = compute_qm_fl_from_row(row)
    elif diagram_key == "Diagram 3 (Q-F-L v2)":
        w = compute_qfl_from_row(row)
    else:
        w = compute_qp_lv_ls_from_row(row)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(img)
    ax.axis("off")

    if any(np.isnan(w)) or sum(w) == 0:
        ax.text(0.5,0.5,"No data for this sample", transform=ax.transAxes, ha="center")
        return fig

    vertex_coords = get_scaled_vertex_coords_for_diagram(diagram_key)
    px, py = barycentric_to_pixel(w, vertex_coords)

    # draw triangle outline for clarity
    tri_x = [vertex_coords[0][0], vertex_coords[1][0], vertex_coords[2][0], vertex_coords[0][0]]
    tri_y = [vertex_coords[0][1], vertex_coords[1][1], vertex_coords[2][1], vertex_coords[0][1]]
    ax.plot(tri_x, tri_y, linestyle="--", linewidth=1, zorder=2)

    ax.scatter(px, py, s=180, color="yellow", edgecolor="k", linewidth=0.9, zorder=10)
    ax.text(px + 10, py - 10, row["Sample"], fontsize=12, weight="bold", zorder=11)

    return fig

# ----------------------
# Create combined plots (one per diagram) showing all samples
# ----------------------
def create_combined_plots(df, outdir):
    files = {}
    for key in DEFAULT_IMG.keys():
        fig, ax = plt.subplots(figsize=(8,8))
        img_arr, scale = DIAGRAM_IMAGES.get(key, (np.ones((600,600,3), dtype=np.uint8), 1.0))
        ax.imshow(img_arr)
        ax.axis("off")

        verts = get_scaled_vertex_coords_for_diagram(key)
        labels = []
        xs = []
        ys = []
        for idx, row in df.iterrows():
            if key == "Diagram 1 (Q-F-L)":
                w = compute_qfl_from_row(row)
            elif key == "Diagram 2 (Qm-F-L)":
                w = compute_qm_fl_from_row(row)
            elif key == "Diagram 3 (Q-F-L v2)":
                w = compute_qfl_from_row(row)
            else:
                w = compute_qp_lv_ls_from_row(row)
            if any(np.isnan(w)) or sum(w) == 0:
                continue
            px, py = barycentric_to_pixel(w, verts)
            xs.append(px); ys.append(py); labels.append(row["Sample"])

        for i,(xx,yy) in enumerate(zip(xs,ys)):
            ax.scatter(xx, yy, s=120, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], edgecolor="k", linewidth=0.6, zorder=11)
            ax.text(xx + 8, yy - 8, labels[i], fontsize=10, weight="bold", zorder=12)

        fname = os.path.join(outdir, f"combined_{key.replace(' ','_').replace('/','_')}_{uuid.uuid4().hex[:6]}.png")
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        files[key] = fname
    return files

# ----------------------
# Build single PDF (all samples). Per sample: write formatted paragraph. After all samples: attach 4 combined plots
# ----------------------
def build_all_samples_pdf(df):
    tmpdir = tempfile.mkdtemp(prefix="pdf_combined_")
    try:
        combined_files = create_combined_plots(df, tmpdir)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("<b>Laporan Analisis Semua Sampel — Gabungan Interpretasi</b>", styles["Title"]))
        story.append(Spacer(1, 12))

        # Per sampel: write 1 paragraph in the requested format
        for idx, row in df.iterrows():
            sample = row["Sample"]

            # compute weights per diagram
            w1 = compute_qfl_from_row(row)       # Diagram 1 -> Pettijohn (Nama Batuan)
            w2 = compute_qm_fl_from_row(row)     # Diagram 2 -> Subzona
            w3 = compute_qfl_from_row(row)       # Diagram 3 -> Zona (uses same compute as diagram1 per your original)
            w4 = compute_qp_lv_ls_from_row(row)  # Diagram 4 -> Provenance

            # labels
            lbl1, _ = interpret_sample(w1, "Diagram 1 (Q-F-L)")
            lbl2, _ = interpret_sample(w2, "Diagram 2 (Qm-F-L)")
            lbl3, _ = interpret_sample(w3, "Diagram 3 (Q-F-L v2)")
            lbl4, _ = interpret_sample(w4, "Diagram 4 (Qp-Lv-Ls)")

            # Header and composition (keep composition short)
            story.append(Paragraph(f"<b>Sampel: {sample}</b>", styles["Heading2"]))
            story.append(Spacer(1,6))
            comp = f"Komposisi input — Qm: {row['Qm']}, Qp: {row['Qp']}, F: {row['F']}, Lv: {row['Lv']}, Ls: {row['Ls']}, Lm: {row['Lm']}"
            story.append(Paragraph(comp, styles["BodyText"]))
            story.append(Spacer(1,6))

            # 1..4 list
            daftar = (
                f"1. Klasifikasi Nama Batuan (Pettijohn, 1987): {lbl1}<br/>"
                f"2. Klasifikasi Subzona Tatanan Tektonik (Dickinson & Suczek, 1979): {lbl2}<br/>"
                f"3. Klasifikasi Zona Tatanan Tektonik (Dickinson & Suczek, 1979): {lbl3}<br/>"
                f"4. Klasifikasi Provenance (Dickinson & Suczek, 1979): {lbl4}<br/>"
            )
            story.append(Paragraph(daftar, styles["BodyText"]))
            story.append(Spacer(1,6))

            # Single paragraph in exact requested format
            paragraf = (
                f"Batu pasir ini adalah <b>{lbl1}</b> (Pettijohn, 1987). "
                f"Berdasarkan jumlah komposisi kuarsa, feldspar, dan lithic, batuan ini diinterpretasikan berasal dari wilayah "
                f"dengan tatanan geologi <b>{lbl3}</b> (Diagram 3), tepatnya pada sub zona <b>{lbl2}</b> (Diagram 2) "
                f"sedangkan sumber material sedimen batuan ini berasal dari <b>{lbl4}</b> (Diagram 4)."
            )
            story.append(Paragraph(paragraf, styles["BodyText"]))
            story.append(Spacer(1,12))

        # Lampiran: combined plots (four diagrams)
        story.append(PageBreak())
        story.append(Paragraph("<b>LAMPIRAN: Plot Gabungan Semua Sampel</b>", styles["Title"]))
        story.append(Spacer(1,12))

        for key in DEFAULT_IMG.keys():
            story.append(Paragraph(f"<b>{key}</b>", styles["Heading3"]))
            imgfile = combined_files.get(key)
            if imgfile and os.path.exists(imgfile):
                try:
                    story.append(RLImage(imgfile, width=15*cm, height=15*cm))
                except Exception as e:
                    story.append(Paragraph(f"Gagal menambahkan gambar {key}: {e}", styles["BodyText"]))
            else:
                story.append(Paragraph("Gambar tidak tersedia.", styles["BodyText"]))
            story.append(Spacer(1,12))

        # Optional overall conclusion
        story.append(PageBreak())
        story.append(Paragraph("<b>Kesimpulan Umum</b>", styles["Heading1"]))
        story.append(Paragraph(
            "Rangkuman: dari sampel yang dianalisis, terdapat variasi provenansi termasuk kontribusi craton (kuarsa mature), "
            "recycled orogen, dan pengaruh busur magmatik pada beberapa sampel.", styles["BodyText"]
        ))

        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    return pdf_bytes

# ----------------------
# Display Streamlit plotting & UI
# ----------------------
st.markdown("### 2) Plot results (lihat & adjust vertex if necessary)")
col1, col2 = st.columns([2,1])

# display image for selected diagram
display_img, display_scale = DIAGRAM_IMAGES.get(diagram_choice, (np.ones((600,600,3), dtype=np.uint8), 1.0))

with col1:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(display_img)
    ax.axis("off")

    # compute vertex coords scaled
    vertex_coords = get_scaled_vertex_coords_for_diagram(diagram_choice)

    # draw triangle outline
    tri_x = [vertex_coords[0][0], vertex_coords[1][0], vertex_coords[2][0], vertex_coords[0][0]]
    tri_y = [vertex_coords[0][1], vertex_coords[1][1], vertex_coords[2][1], vertex_coords[0][1]]
    ax.plot(tri_x, tri_y, linestyle="--", linewidth=1, zorder=2)

    for i, row in edited.iterrows():
        if diagram_choice == "Diagram 1 (Q-F-L)":
            w = compute_qfl_from_row(row)
        elif diagram_choice == "Diagram 2 (Qm-F-L)":
            w = compute_qm_fl_from_row(row)
        elif diagram_choice == "Diagram 3 (Q-F-L v2)":
            w = compute_qfl_from_row(row)
        else:
            w = compute_qp_lv_ls_from_row(row)

        if any(np.isnan(w)) or sum(w) == 0:
            continue
        px, py = barycentric_to_pixel(w, vertex_coords)
        ax.scatter(px, py, s=140, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], edgecolor="k", linewidth=0.7, zorder=10)
        ax.text(px + 10, py - 10, row["Sample"], fontsize=11, weight="bold", zorder=11)

    st.pyplot(fig)

with col2:
    st.markdown("#### Vertex coords (original pixel-space, scaled to image)")
    st.write({"v1": (v1x, v1y), "v2": (v2x, v2y), "v3": (v3x, v3y)})

    st.markdown("#### Computed percentages & classification (current diagram)")
    preview = []
    for i, row in edited.iterrows():
        if diagram_choice == "Diagram 1 (Q-F-L)":
            w = compute_qfl_from_row(row)
            cls, _ = interpret_sample(w, diagram_choice)
        elif diagram_choice == "Diagram 2 (Qm-F-L)":
            w = compute_qm_fl_from_row(row)
            cls, _ = interpret_sample(w, diagram_choice)
        elif diagram_choice == "Diagram 3 (Q-F-L v2)":
            w = compute_qfl_from_row(row)
            cls, _ = interpret_sample(w, diagram_choice)
        else:
            w = compute_qp_lv_ls_from_row(row)
            cls, _ = interpret_sample(w, diagram_choice)
        preview.append({"Sample": row["Sample"], "W1 %": round(w[0],2), "W2 %": round(w[1],2), "W3 %": round(w[2],2), "Class": cls})
    st.table(pd.DataFrame(preview))

    st.markdown("---")
    if st.button("Generate & Download Single PDF (All samples combined)"):
        st.info("Menyusun PDF... (membuat plot gabungan lalu menyusun PDF)")
        pdf_bytes = build_all_samples_pdf(edited)
        st.download_button("Download PDF (All samples)", data=pdf_bytes, file_name="all_samples_combined.pdf", mime="application/pdf")

# also allow download of current plot as PNG
png_buf = BytesIO()
fig.savefig(png_buf, format="png", bbox_inches="tight", dpi=150)
png_buf.seek(0)
st.download_button("Download current diagram (PNG)", data=png_buf, file_name="current_diagram.png", mime="image/png")

st.markdown("---")
st.caption("Notes: interpretasi gabungan di PDF dibuat sesuai format yang diminta. Pada lampiran disertakan 4 plot gabungan yang menampilkan posisi semua sampel pada masing-masing diagram.")
