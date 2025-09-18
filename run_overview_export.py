import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from odf.opendocument import OpenDocumentPresentation
from odf.draw import Page, Frame, Image as DrawImage, TextBox, Line, G
from odf.text import P
from odf.style import Style, GraphicProperties, TextProperties, PageLayout, PageLayoutProperties, MasterPage
from PIL import Image
from typing import List, Dict, Tuple


# ---------------------------
# Image processing
# ---------------------------

def plane_correction(img: np.ndarray) -> np.ndarray:
    """
    Subtract best-fit plane from image (least-squares).
    """
    h, w = img.shape
    y, x = np.mgrid[:h, :w]
    xc, yc = x - x.mean(), y - y.mean()
    z = img

    a = np.array([
        [np.sum(xc * xc), np.sum(xc * yc), np.sum(xc)],
        [np.sum(yc * xc), np.sum(yc * yc), np.sum(yc)],
        [np.sum(xc), np.sum(yc), h * w]
    ])
    b = np.array([np.sum(xc * z), np.sum(yc * z), np.sum(z)])

    coeff = np.linalg.solve(a, b)
    plane = coeff[0] * xc + coeff[1] * yc + coeff[2]
    return img - plane


def _load_xyz_data(xyz_file: str, entry: Dict[str, str]) -> np.ndarray:
    """
    Load XYZ file and return plane-corrected 2D array.
    """
    clean_lines = []
    with open(xyz_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or line.startswith("WSxM") or line.strip().startswith("X["):
                continue
            clean_lines.append(line)

    data = np.loadtxt(clean_lines)
    zs = data[:, 2]
    width, height = int(entry["Width"]), int(entry["Height"])
    img = zs.reshape((height, width))
    return plane_correction(img)


def _save_image_png(img: np.ndarray, base_file: str) -> str:
    """
    Save corrected image as PNG.
    """
    vmin, vmax = np.nanmin(img), np.nanmax(img)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.imshow(img, cmap="cividis", origin="lower", vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.tight_layout()
    img_file = base_file + "_img.png"
    plt.savefig(img_file, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return img_file


def _save_colorbar_png(img: np.ndarray, base_file: str, unit: str) -> str:
    """
    Save colorbar as PNG.
    """
    vmin, vmax = np.nanmin(img), np.nanmax(img)
    fig, ax = plt.subplots(figsize=(6, 0.5), dpi=300)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="cividis"),
        cax=ax, orientation="horizontal", label=unit
    )
    cbar_file = base_file + "_cbar.png"
    plt.savefig(cbar_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return cbar_file


def xyz_to_pngs(xyz_file: str, base_file: str, entry: Dict[str, str]) -> Tuple[str, str]:
    """
    Convert XYZ file to image PNG and colorbar PNG.
    """
    img = _load_xyz_data(xyz_file, entry)
    img_file = _save_image_png(img, base_file)
    cbar_file = _save_colorbar_png(img, base_file, entry.get("Unit", "a.u."))
    return img_file, cbar_file


# ---------------------------
# Manifest
# ---------------------------

def load_manifest(manifest_file: str) -> List[Dict[str, str]]:
    """
    Load CSV manifest into list of dicts.
    """
    with open(manifest_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sort_manifest(manifest: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Sort manifest entries by datetime.
    """
    for entry in manifest:
        try:
            dt = datetime.strptime(entry["Date"] + " " + entry["Time"], "%m/%d/%Y %H:%M:%S")
        except Exception:
            dt = datetime.max
        entry["_datetime"] = dt
    return sorted(manifest, key=lambda e: e["_datetime"])


# ---------------------------
# Slide building helpers
# ---------------------------

def _add_scalebar(doc, group, entry, left_cm: float, top_cm: float, slot_width: float):
    """
    Add scalebar to group of slide elements.
    """
    scan_x, scan_y = float(entry["ScanX"]), float(entry["ScanY"])
    abs_size_nm = max(scan_x, scan_y)
    steps = [1, 2, 3, 4, 5, 10, 20, 25, 50, 100]
    step = max([s for s in steps if s <= abs_size_nm / 3])

    linestyle = Style(name="scalebar", family="graphic")
    linestyle.addElement(GraphicProperties(stroke="solid", strokecolor="#ffffff", strokewidth="0.1cm"))
    doc.styles.addElement(linestyle)

    bar_len_cm = (step / scan_x) * slot_width
    line = Line(
        stylename=linestyle,
        x1=f"{left_cm+1}cm", y1=f"{top_cm+slot_width-0.7}cm",
        x2=f"{left_cm+1+bar_len_cm}cm", y2=f"{top_cm+slot_width-0.7}cm"
    )
    group.addElement(line)

    lab_frame = Frame(width="3cm", height="1cm",
                      x=f"{left_cm+1}cm", y=f"{top_cm+slot_width-1.4}cm")
    tb = TextBox()
    tb.addElement(P(text=f"{step} nm"))
    lab_frame.addElement(tb)
    group.addElement(lab_frame)


def _add_caption(doc, slide, entry, left_cm: float, top_cm: float, slot_width: float):
    """
    Add caption below grouped image.
    """
    caption_style = Style(name="captionstyle", family="paragraph")
    caption_style.addElement(TextProperties(fontname="Arial", fontsize="10pt"))
    doc.styles.addElement(caption_style)

    cap_frame = Frame(width=f"{slot_width}cm", height="2cm",
                      x=f"{left_cm}cm", y=f"{top_cm+slot_width+1.2}cm")
    tb = TextBox()
    fname = entry["File"][:32] + "..." if len(entry["File"]) > 35 else entry["File"]
    tb.addElement(P(stylename=caption_style, text=f"□ {fname} ({entry['Date']} {entry['Time']})"))
    tb.addElement(P(stylename=caption_style,
                    text=(f"{entry['Channel']} | {entry['Width']}×{entry['Height']} px | "
                          f"{entry['ScanX']}×{entry['ScanY']} {entry.get('Unit','')} | "
                          f"{entry.get('Bias','')} {entry.get('BiasPhysUnit','')} | "
                          f"{entry.get('SetPoint','')} {entry.get('SetPointPhysUnit','')}")))
    cap_frame.addElement(tb)
    slide.addElement(cap_frame)


def add_grouped_image(
    slide, doc, img_file: str, cbar_file: str, entry: Dict[str, str],
    left_cm: float, top_cm: float, slot_width: float, slot_height: float
):
    """
    Add grouped image (main, scalebar, colorbar, caption) to slide.
    """
    group = G()

    # main image
    img_frame = Frame(width=f"{slot_width}cm", height=f"{slot_width}cm",
                      x=f"{left_cm}cm", y=f"{top_cm}cm")
    href = doc.addPicture(img_file)
    img_frame.addElement(DrawImage(href=href))
    group.addElement(img_frame)

    # scalebar
    _add_scalebar(doc, group, entry, left_cm, top_cm, slot_width)

    # colorbar
    cbar_frame = Frame(width=f"{slot_width*0.8}cm", height="0.6cm",
                       x=f"{left_cm+1}cm", y=f"{top_cm+slot_width+0.3}cm")
    href = doc.addPicture(cbar_file)
    cbar_frame.addElement(DrawImage(href=href))
    group.addElement(cbar_frame)

    slide.addElement(group)

    # caption
    _add_caption(doc, slide, entry, left_cm, top_cm, slot_width)


# ---------------------------
# Main ODP export
# ---------------------------

def xyz_folder_to_odp(folder: str, manifest_file: str, out_odp: str = "overview.odp"):
    """
    Build ODP overview presentation from a folder of .xyz files and a manifest.
    """
    if not os.path.exists(manifest_file):
        print(f"⚠️ Manifest not found: {manifest_file}")
        return

    manifest = sort_manifest(load_manifest(manifest_file))
    doc = OpenDocumentPresentation()

    # square layout
    pagelayout = PageLayout(name="Square")
    pagelayout.addElement(PageLayoutProperties(pagewidth="25cm", pageheight="25cm"))
    doc.automaticstyles.addElement(pagelayout)
    doc.masterstyles.addElement(MasterPage(name="Default", pagelayoutname=pagelayout))

    # title slide
    slide = Page(masterpagename="Default", name="Title")
    frame = Frame(width="20cm", height="5cm", x="2cm", y="2cm")
    tb = TextBox()
    tb.addElement(P(text="SXM Export Overview"))
    tb.addElement(P(text=f"Folder: {os.path.abspath(folder)}"))
    tb.addElement(P(text=f"{len(manifest)} exported channels"))
    frame.addElement(tb)
    slide.addElement(frame)
    doc.presentation.addElement(slide)

    # grid layout
    slot_width, slot_height = 10, 12
    positions = [(2 + col * 12, 0.5 + row * 12) for row in range(2) for col in range(2)]

    for i, entry in enumerate(tqdm(manifest, desc="Building ODP", unit="file"), start=1):
        if (i - 1) % 4 == 0:
            slide = Page(masterpagename="Default", name=f"Slide {i}")
            doc.presentation.addElement(slide)

        idx = (i - 1) % 4
        left_cm, top_cm = positions[idx]

        xyz_file = os.path.join(folder, entry["File"])
        if not os.path.exists(xyz_file):
            print(f"⚠️ Missing xyz file: {xyz_file}")
            continue

        base_file = xyz_file.replace(".xyz", "")
        img_file, cbar_file = xyz_to_pngs(xyz_file, base_file, entry)
        add_grouped_image(slide, doc, img_file, cbar_file, entry, left_cm, top_cm, slot_width, slot_height)

    doc.save(out_odp)
    print(f"\n✔ ODP overview saved to {out_odp}")


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "data")
    manifest_file = os.path.join(base_path, "valid_files_log.csv")
    out_odp = os.path.join(base_path, "overview.odp")
    xyz_folder_to_odp(base_path, manifest_file, out_odp)
