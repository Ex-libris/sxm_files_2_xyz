import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from datetime import datetime
from odf.opendocument import OpenDocumentPresentation
from odf.draw import Page, Frame, Image as DrawImage, TextBox, Line, G
from odf.text import P
from odf.style import Style, GraphicProperties, TextProperties, PageLayout, PageLayoutProperties, MasterPage
from PIL import Image


def plane_correction(img):
    """Subtract best-fit plane from image (quick least-squares)."""
    h, w = img.shape
    Y, X = np.mgrid[:h, :w]
    Xc, Yc = X - X.mean(), Y - Y.mean()
    Z = img
    A = np.array([
        [np.sum(Xc * Xc), np.sum(Xc * Yc), np.sum(Xc)],
        [np.sum(Yc * Xc), np.sum(Yc * Yc), np.sum(Yc)],
        [np.sum(Xc), np.sum(Yc), h * w]
    ])
    b = np.array([np.sum(Xc * Z), np.sum(Yc * Z), np.sum(Z)])
    coeff = np.linalg.solve(A, b)
    plane = coeff[0] * Xc + coeff[1] * Yc + coeff[2]
    return img - plane


def xyz_to_pngs(xyz_file, base_file, entry):
    """Make two PNGs: one for image, one for colorbar only."""
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
    img_corr = plane_correction(img)

    vmin, vmax = np.nanmin(img_corr), np.nanmax(img_corr)

    # main image
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    im = ax.imshow(img_corr, cmap="cividis", origin="lower", vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.tight_layout()
    img_file = base_file + "_img.png"
    plt.savefig(img_file, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # colorbar
    fig, ax = plt.subplots(figsize=(6, 0.5), dpi=300)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="cividis"),
        cax=ax, orientation="horizontal", label=entry.get("Unit", "a.u.")
    )
    cbar_file = base_file + "_cbar.png"
    plt.savefig(cbar_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return img_file, cbar_file


def load_manifest(manifest_file):
    with open(manifest_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sort_manifest(manifest):
    for entry in manifest:
        try:
            dt = datetime.strptime(entry["Date"] + " " + entry["Time"], "%m/%d/%Y %H:%M:%S")
        except Exception:
            dt = datetime.max
        entry["_datetime"] = dt
    return sorted(manifest, key=lambda e: e["_datetime"])


def add_grouped_image(slide, doc, img_file, cbar_file, entry, left_cm, top_cm, slot_width, slot_height):
    group = G()

    # image frame
    img_frame = Frame(width=f"{slot_width}cm", height=f"{slot_width}cm",
                      x=f"{left_cm}cm", y=f"{top_cm}cm")
    href = doc.addPicture(img_file)
    img_frame.addElement(DrawImage(href=href))
    group.addElement(img_frame)

    # scale bar calc
    scan_x, scan_y = float(entry["ScanX"]), float(entry["ScanY"])
    abs_size_nm = max(scan_x, scan_y)
    steps = [1, 2, 3, 4, 5, 10, 20, 25, 50, 100]
    step = max([s for s in steps if s <= abs_size_nm / 3])

    linestyle = Style(name="scalebar", family="graphic")
    linestyle.addElement(GraphicProperties(stroke="solid", strokecolor="#ffffff", strokewidth="0.1cm"))
    doc.styles.addElement(linestyle)

    bar_len_cm = (step / scan_x) * slot_width
    line = Line(stylename=linestyle,
                x1=f"{left_cm+1}cm", y1=f"{top_cm+slot_width-0.7}cm",
                x2=f"{left_cm+1+bar_len_cm}cm", y2=f"{top_cm+slot_width-0.7}cm")
    group.addElement(line)

    # size label above line
    lab_frame = Frame(width="3cm", height="1cm",
                      x=f"{left_cm+1}cm", y=f"{top_cm+slot_width-1.4}cm")
    tb = TextBox(); tb.addElement(P(text=f"{step} nm"))
    lab_frame.addElement(tb)
    group.addElement(lab_frame)

    # colorbar
    cbar_frame = Frame(width=f"{slot_width*0.8}cm", height="0.6cm",
                       x=f"{left_cm+1}cm", y=f"{top_cm+slot_width+0.3}cm")
    href = doc.addPicture(cbar_file)
    cbar_frame.addElement(DrawImage(href=href))
    group.addElement(cbar_frame)

    slide.addElement(group)

    # caption
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


def xyz_folder_to_odp(folder, manifest_file, out_odp="overview.odp"):
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

    # grid: 2×2
    slot_width, slot_height = 10, 12
    positions = [(2 + col * 12,0.5 + row * 12) for row in range(2) for col in range(2)]

    for i, entry in enumerate(tqdm(manifest, desc="Building ODP", unit="file"), start=1):
        if (i - 1) % 4 == 0:
            slide = Page(masterpagename="Default", name=f"Slide {i}")
            doc.presentation.addElement(slide)

        idx = (i - 1) % 4
        left_cm, top_cm = positions[idx]

        xyz_file = os.path.join(folder, entry["File"])
        if not os.path.exists(xyz_file):
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
