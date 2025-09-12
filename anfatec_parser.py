import os
import numpy as np
from datetime import datetime

# ---------------------------
# Parsing
# ---------------------------

def parse_anfatec_txt(txt_file):
    global_params = {}
    channels = []

    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # global params
    for line in lines:
        if ":" in line and not line.strip().startswith(";"):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.split(";")[0].strip()
            global_params[key] = val

    # channel blocks
    block = None
    for line in lines:
        if line.strip().startswith("FileDescBegin"):
            block = {}
        elif line.strip().startswith("FileDescEnd"):
            if block:
                channels.append(block)
            block = None
        elif block is not None and ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.split(";")[0].strip()
            block[key] = val

    return global_params, channels, lines


# ---------------------------
# Timestamp handling
# ---------------------------

def parse_timestamp(raw_date, raw_time):
    fmts = [
        "%m/%d/%Y %I:%M:%S %p",  # US AM/PM
        "%m/%d/%Y %H:%M:%S",     # US 24h
        "%d/%m/%Y %H:%M:%S",     # EU style
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(f"{raw_date} {raw_time}", fmt)
        except ValueError:
            continue
    return None


# ---------------------------
# Fast plane correction
# ---------------------------

def plane_correction(img):
    """Subtract best-fit plane from image (fast closed-form)."""
    h, w = img.shape
    Y, X = np.mgrid[:h, :w]

    Xc = X - X.mean()
    Yc = Y - Y.mean()
    Z = img

    A = np.array([
        [np.sum(Xc*Xc), np.sum(Xc*Yc), np.sum(Xc)],
        [np.sum(Yc*Xc), np.sum(Yc*Yc), np.sum(Yc)],
        [np.sum(Xc),    np.sum(Yc),    h*w]
    ])
    b = np.array([
        np.sum(Xc*Z),
        np.sum(Yc*Z),
        np.sum(Z)
    ])

    coeff = np.linalg.solve(A, b)
    plane = coeff[0]*Xc + coeff[1]*Yc + coeff[2]
    return img - plane


# ---------------------------
# Active area fraction
# ---------------------------

def active_area_fraction(img, var_threshold=1e-12):
    """Fraction of rows with variance above threshold (proxy for valid acquired area)."""
    row_vars = np.nanstd(img, axis=1)
    active_rows = np.sum(row_vars > var_threshold)
    return active_rows / img.shape[0]


# ---------------------------
# Signal validation
# ---------------------------

def is_valid_signal(data, width, height, var_threshold=1e-12, row_threshold=0.05, lazy_ratio=50):
    """Check if a signal contains meaningful information."""
    meta = {"active_area_fraction": None}

    # trivial checks
    if np.all(np.isnan(data)):
        return False, "all_nan", meta
    if np.allclose(data, 0, atol=1e-12):
        return False, "all_zeros", meta

    unique_vals = np.unique(data)
    if len(unique_vals) <= 2:
        return False, f"too_few_unique_values ({len(unique_vals)})", meta

    img = data.reshape((height, width))

    # active area fraction
    frac_active = active_area_fraction(img, var_threshold)
    meta["active_area_fraction"] = frac_active
    if frac_active < 0.1:  # less than 10% rows carry info
        return False, f"too_little_acquired_area ({frac_active*100:.1f}% active)", meta

    # global variance
    raw_var = np.nanstd(img)
    if raw_var < var_threshold:
        return False, "low_variance_raw", meta

    # row checks
    row_vars = np.nanstd(img, axis=1)
    flat_rows = np.sum(row_vars < var_threshold)
    frac_flat = flat_rows / height
    if frac_flat > (1 - row_threshold):
        return False, f"mostly_flat_rows ({frac_flat*100:.1f}% flat)", meta

    # row detrend variance
    row_detrended = img - img.mean(axis=1, keepdims=True)
    detrended_var = np.nanstd(row_detrended)
    if detrended_var < var_threshold * 10:
        return False, "no_spatial_structure_after_row_detrend", meta

    # gradient variance
    gx, gy = np.gradient(img)
    grad_var = np.nanstd(gx) + np.nanstd(gy)
    if grad_var < var_threshold * 100:
        return False, "low_gradient_variance", meta

    # plane correction
    if raw_var < lazy_ratio * var_threshold:
        img_corr = plane_correction(img)
        corr_var = np.nanstd(img_corr)
        if corr_var < var_threshold:
            return False, "low_variance_after_plane_correction", meta

    return True, "valid", meta


# ---------------------------
# Main export function
# ---------------------------

def anfatec_to_wsxm(txt_file, base_path=""):
    txt_path = os.path.join(base_path, txt_file)
    global_params, channels, raw_txt_lines = parse_anfatec_txt(txt_path)

    # image size
    height = int(global_params["yPixel"])
    width = int(global_params["xPixel"])

    # scan size and units
    scan_x = float(global_params["XScanRange"])
    scan_y = float(global_params["YScanRange"])
    unit_x = global_params.get("XPhysUnit", "µm")
    unit_y = global_params.get("YPhysUnit", "µm")

    # timestamp
    raw_date = global_params.get("Date", "")
    raw_time = global_params.get("Time", "")
    dt = parse_timestamp(raw_date, raw_time)
    ts_str = dt.strftime("%Y%m%d_%H%M%S") if dt else "unknowntime"

    basename = os.path.splitext(os.path.basename(txt_file))[0]

    exports = []
    invalid_exports = []

    # Detect mode (constant current vs constant height)
    topo_channels = [ch for ch in channels if "topo" in ch.get("Caption", "").lower()]
    mode = None

    if topo_channels:
        topo_valid_any = False
        for ch in topo_channels:
            int_path = os.path.join(base_path, ch["FileName"])
            if os.path.exists(int_path):
                data = np.memmap(int_path, dtype=np.int32, mode="r")
                valid, _, _ = is_valid_signal(data, width, height)
                if valid:
                    topo_valid_any = True
        if topo_valid_any:
            mode = "constant_current"
        else:
            mode = "constant_height"
    else:
        mode = "constant_height"

    # Export channels with filtering
    for ch in channels:
        fname = ch["FileName"]
        caption = ch.get("Caption", fname)
        caption_lower = caption.lower()
        scale = float(ch["Scale"])
        unit_z = ch.get("PhysUnit", "a.u.")
        offset = float(ch.get("Offset", 0))

        # skip backward channels
        if "bwd" in caption_lower:
            invalid_exports.append({"File": fname, "Channel": caption,
                                    "Status": "invalid", "Reason": "backward_channel",
                                    "Mode": mode, "ActiveFraction": "NA", "Cropped": "NA"})
            continue

        # mode-based filtering
        if mode == "constant_current" and "topo" not in caption_lower:
            invalid_exports.append({"File": fname, "Channel": caption,
                                    "Status": "invalid", "Reason": "not_relevant_in_constant_current",
                                    "Mode": mode, "ActiveFraction": "NA", "Cropped": "NA"})
            continue
        if mode == "constant_height" and "topo" in caption_lower:
            invalid_exports.append({"File": fname, "Channel": caption,
                                    "Status": "invalid", "Reason": "topography_flat_in_constant_height",
                                    "Mode": mode, "ActiveFraction": "NA", "Cropped": "NA"})
            continue

        int_path = os.path.join(base_path, fname)
        if not os.path.exists(int_path):
            invalid_exports.append({"File": fname, "Channel": caption,
                                    "Status": "invalid", "Reason": "missing_file",
                                    "Mode": mode, "ActiveFraction": "NA", "Cropped": "NA"})
            continue

        data = np.memmap(int_path, dtype=np.int32, mode="r")
        if data.size != width * height:
            invalid_exports.append({"File": fname, "Channel": caption,
                                    "Status": "invalid", "Reason": "size_mismatch",
                                    "Mode": mode, "ActiveFraction": "NA", "Cropped": "NA"})
            continue

        valid, reason, meta = is_valid_signal(data, width, height)

        active_frac = meta.get("active_area_fraction")
        active_frac_str = f"{active_frac:.3f}" if active_frac is not None else "NA"

        if not valid:
            invalid_exports.append({"File": fname, "Channel": caption,
                                    "Status": "invalid", "Reason": reason,
                                    "Mode": mode,
                                    "ActiveFraction": active_frac_str,
                                    "Cropped": "NA"})
            continue

        # Passed validation → apply cropping
        img = data.reshape((height, width)) * scale + offset
        row_vars = np.nanstd(img, axis=1)
        active = row_vars > 1e-12

        cropped = "No"
        if np.any(active):
            y_min = np.argmax(active)
            y_max = len(active) - np.argmax(active[::-1])
            if y_min > 0 or y_max < height:
                img = img[y_min:y_max, :]
                cropped = "Yes"

        height_cropped = img.shape[0]

        # grids
        xs = np.linspace(0, scan_x, width, endpoint=False)
        ys = np.linspace(0, scan_y * height_cropped / height, height_cropped, endpoint=False)
        XX, YY = np.meshgrid(xs, ys)

        # points
        points = np.column_stack([XX.ravel(), YY.ravel(), img.ravel()])

        # output
        safe_caption = caption.replace(" ", "_")
        out_file = os.path.join(base_path, f"{basename}_{safe_caption}_{ts_str}.xyz")

        header_lines = [
            "WSxM file copyright UAM",
            "WSxM ASCII XYZ file",
            f"X[{unit_x}]\tY[{unit_y}]\tZ[{unit_z}]",
            "# --- Original Anfatec header ---"
        ] + ["# " + line.rstrip() for line in raw_txt_lines] + ["# --- End of original header ---"]
        header_str = "\n".join(header_lines)

        np.savetxt(out_file, points, fmt="%.6f\t%.6f\t%.8f",
                   header=header_str, comments="")

        exports.append({
            "File": os.path.basename(out_file),
            "Channel": caption,
            "Scale": scale,
            "Unit": unit_z,
            "Width": width,
            "Height": height_cropped,
            "ScanX": scan_x,
            "ScanY": scan_y * height_cropped / height,
            "User": global_params.get("UserName", ""),
            "Date": global_params.get("Date", ""),
            "Time": global_params.get("Time", ""),
            "System": global_params.get("SystemType", ""),
            "Mode": mode,
            "ActiveFraction": active_frac_str,
            "Cropped": cropped,
            "Bias": global_params.get("Bias", ""),
            "BiasPhysUnit": global_params.get("BiasPhysUnit", ""),
            "SetPoint": global_params.get("SetPoint", ""),
            "SetPointPhysUnit": global_params.get("SetPointPhysUnit", "")
        })

    return exports, invalid_exports
