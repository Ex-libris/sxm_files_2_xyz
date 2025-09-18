import os
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Parsing
# ---------------------------

def parse_anfatec_txt(
    txt_file: str
) -> Tuple[Dict[str, str], List[Dict[str, str]], List[str]]:
    """
    Parse an Anfatec .txt file.

    Parameters
    ----------
    txt_file : str
        Path to the text file.

    Returns
    -------
    global_params : dict
        Global parameters from header.
    channels : list of dict
        Channel metadata blocks.
    lines : list of str
        Raw lines from the file.
    """
    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    global_params = _parse_global_params(lines)
    channels = _parse_channel_blocks(lines)
    return global_params, channels, lines


def _parse_global_params(lines: List[str]) -> Dict[str, str]:
    """Extract global parameters from text header lines."""
    params: Dict[str, str] = {}
    for line in lines:
        if ":" in line and not line.strip().startswith(";"):
            key, val = line.split(":", 1)
            params[key.strip()] = val.split(";")[0].strip()
    return params


def _parse_channel_blocks(lines: List[str]) -> List[Dict[str, str]]:
    """Extract channel metadata blocks from text lines."""
    channels: List[Dict[str, str]] = []
    block: Optional[Dict[str, str]] = None
    for line in lines:
        if line.strip().startswith("FileDescBegin"):
            block = {}
        elif line.strip().startswith("FileDescEnd"):
            if block:
                channels.append(block)
            block = None
        elif block is not None and ":" in line:
            key, val = line.split(":", 1)
            block[key.strip()] = val.split(";")[0].strip()
    return channels


# ---------------------------
# Timestamp handling
# ---------------------------

def parse_timestamp(raw_date: str, raw_time: str) -> Optional[datetime]:
    """
    Parse date and time strings into a datetime object.

    Tries multiple formats: US AM/PM, US 24h, EU 24h.

    Returns
    -------
    datetime or None
        Parsed datetime, or None if all formats fail.
    """
    fmts = [
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(f"{raw_date} {raw_time}", fmt)
        except ValueError:
            continue
    return None


# ---------------------------
# Image utilities
# ---------------------------

def plane_correction(img: np.ndarray) -> np.ndarray:
    """
    Subtract best-fit plane from image.

    Parameters
    ----------
    img : ndarray
        Input 2D array.

    Returns
    -------
    ndarray
        Plane-corrected image.
    """
    h, w = img.shape
    y, x = np.mgrid[:h, :w]

    xc = x - x.mean()
    yc = y - y.mean()
    z = img

    a = np.array([
        [np.sum(xc * xc), np.sum(xc * yc), np.sum(xc)],
        [np.sum(yc * xc), np.sum(yc * yc), np.sum(yc)],
        [np.sum(xc), np.sum(yc), h * w]
    ])
    b = np.array([
        np.sum(xc * z),
        np.sum(yc * z),
        np.sum(z)
    ])

    coeff = np.linalg.solve(a, b)
    plane = coeff[0] * xc + coeff[1] * yc + coeff[2]
    return img - plane


def active_area_fraction(img: np.ndarray, var_threshold: float = 1e-12) -> float:
    """
    Compute fraction of rows with variance above threshold.

    Parameters
    ----------
    img : ndarray
        2D array.
    var_threshold : float
        Minimum variance to consider a row active.

    Returns
    -------
    float
        Fraction of active rows.
    """
    row_vars = np.nanstd(img, axis=1)
    active_rows = np.sum(row_vars > var_threshold)
    return active_rows / img.shape[0]


# ---------------------------
# Signal validation
# ---------------------------

def is_valid_signal(
    data: np.ndarray,
    width: int,
    height: int,
    var_threshold: float = 1e-12,
    row_threshold: float = 0.05,
    lazy_ratio: float = 50
) -> Tuple[bool, str, Dict[str, Optional[float]]]:
    """
    Check if a signal contains useful information.

    Parameters
    ----------
    data : ndarray
        Flattened signal.
    width : int
        Image width.
    height : int
        Image height.
    var_threshold : float
        Minimum variance threshold.
    row_threshold : float
        Max fraction of flat rows allowed.
    lazy_ratio : float
        Scaling factor for plane correction check.

    Returns
    -------
    valid : bool
        True if signal passes checks.
    reason : str
        Reason if invalid.
    meta : dict
        Extra metrics (e.g. active area fraction).
    """
    meta: Dict[str, Optional[float]] = {"active_area_fraction": None}

    if np.all(np.isnan(data)):
        return False, "all_nan", meta
    if np.allclose(data, 0, atol=1e-12):
        return False, "all_zeros", meta
    if len(np.unique(data)) <= 2:
        return False, "too_few_unique_values", meta

    img = data.reshape((height, width))
    frac_active = active_area_fraction(img, var_threshold)
    meta["active_area_fraction"] = frac_active
    if frac_active < 0.1:
        return False, "too_little_acquired_area", meta

    raw_var = np.nanstd(img)
    if raw_var < var_threshold:
        return False, "low_variance_raw", meta

    if _too_many_flat_rows(img, height, var_threshold, row_threshold):
        return False, "mostly_flat_rows", meta

    if _no_spatial_structure(img, var_threshold):
        return False, "no_spatial_structure_after_row_detrend", meta

    if _low_gradient_variance(img, var_threshold):
        return False, "low_gradient_variance", meta

    if _fails_plane_correction(img, raw_var, var_threshold, lazy_ratio):
        return False, "low_variance_after_plane_correction", meta

    return True, "valid", meta


def _too_many_flat_rows(
    img: np.ndarray, height: int, var_threshold: float, row_threshold: float
) -> bool:
    """Check if most rows are flat below variance threshold."""
    row_vars = np.nanstd(img, axis=1)
    flat_rows = np.sum(row_vars < var_threshold)
    return (flat_rows / height) > (1 - row_threshold)


def _no_spatial_structure(img: np.ndarray, var_threshold: float) -> bool:
    """Check if row-detrended variance is too small."""
    row_detrended = img - img.mean(axis=1, keepdims=True)
    return np.nanstd(row_detrended) < var_threshold * 10


def _low_gradient_variance(img: np.ndarray, var_threshold: float) -> bool:
    """Check if gradient variance is too small."""
    gx, gy = np.gradient(img)
    return (np.nanstd(gx) + np.nanstd(gy)) < var_threshold * 100


def _fails_plane_correction(
    img: np.ndarray, raw_var: float, var_threshold: float, lazy_ratio: float
) -> bool:
    """Check if variance stays low after plane correction."""
    if raw_var < lazy_ratio * var_threshold:
        img_corr = plane_correction(img)
        return np.nanstd(img_corr) < var_threshold
    return False


# ---------------------------
# Export helpers
# ---------------------------

def detect_mode(
    channels: List[Dict[str, str]], base_path: str, width: int, height: int
) -> str:
    """
    Detect scan mode.

    Returns
    -------
    str
        "constant_current" if a topo channel is valid,
        otherwise "constant_height".
    """
    topo_channels = [ch for ch in channels if "topo" in ch.get("Caption", "").lower()]
    if not topo_channels:
        return "constant_height"

    for ch in topo_channels:
        int_path = os.path.join(base_path, ch["FileName"])
        if os.path.exists(int_path):
            data = np.memmap(int_path, dtype=np.int32, mode="r")
            valid, _, _ = is_valid_signal(data, width, height)
            if valid:
                return "constant_current"
    return "constant_height"


def filter_channel(ch: Dict[str, str], mode: str) -> Optional[str]:
    """
    Decide if a channel should be skipped.

    Returns
    -------
    str or None
        Reason for skipping, or None if acceptable.
    """
    caption = ch.get("Caption", ch["FileName"]).lower()
    if "bwd" in caption:
        return "backward_channel"
    if mode == "constant_current" and "topo" not in caption:
        return "not_relevant_in_constant_current"
    if mode == "constant_height" and "topo" in caption:
        return "topography_flat_in_constant_height"
    return None


def validate_channel(
    ch: Dict[str, str], base_path: str, width: int, height: int
) -> Tuple[Optional[np.ndarray], str, str]:
    """
    Validate one channel by loading data and checking signal.

    Returns
    -------
    img : ndarray or None
        Reshaped image if valid.
    status : str
        "valid" or reason string.
    active_frac_str : str
        Active fraction string.
    """
    fname = ch["FileName"]
    int_path = os.path.join(base_path, fname)

    if not os.path.exists(int_path):
        return None, "missing_file", "NA"

    data = np.memmap(int_path, dtype=np.int32, mode="r")
    if data.size != width * height:
        return None, "size_mismatch", "NA"

    valid, reason, meta = is_valid_signal(data, width, height)
    active_frac = meta.get("active_area_fraction")
    active_frac_str = f"{active_frac:.3f}" if active_frac is not None else "NA"

    if not valid:
        return None, reason, active_frac_str

    return data.reshape((height, width)), "valid", active_frac_str


def export_channel(
    fname: str,
    caption: str,
    img: np.ndarray,
    scale: float,
    offset: float,
    width: int,
    height: int,
    scan_x: float,
    scan_y: float,
    unit_x: str,
    unit_y: str,
    unit_z: str,
    global_params: Dict[str, str],
    raw_txt_lines: List[str],
    basename: str,
    ts_str: str,
    active_frac_str: str,
    mode: str,
    base_path: str
) -> Dict[str, Any]:
    """
    Export one channel to WSxM .xyz file.

    Returns
    -------
    dict
        Metadata for the exported file.
    """
    img = img * scale + offset
    cropped, img = _crop_image(img)
    height_cropped = img.shape[0]

    xs = np.linspace(0, scan_x, width, endpoint=False)
    ys = np.linspace(0, scan_y * height_cropped / height, height_cropped, endpoint=False)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel(), img.ravel()])

    safe_caption = caption.replace(" ", "_")
    out_file = os.path.join(base_path, f"{basename}_{safe_caption}_{ts_str}.xyz")

    header_str = _build_header(raw_txt_lines, unit_x, unit_y, unit_z)
    np.savetxt(out_file, points, fmt="%.6f\t%.6f\t%.8f", header=header_str, comments="")

    return {
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
    }


def _crop_image(img: np.ndarray) -> Tuple[str, np.ndarray]:
    """Crop inactive rows from image."""
    row_vars = np.nanstd(img, axis=1)
    active = row_vars > 1e-12
    if not np.any(active):
        return "No", img
    y_min = np.argmax(active)
    y_max = len(active) - np.argmax(active[::-1])
    if y_min > 0 or y_max < img.shape[0]:
        return "Yes", img[y_min:y_max, :]
    return "No", img


def _build_header(raw_txt_lines: List[str], unit_x: str, unit_y: str, unit_z: str) -> str:
    """Build WSxM header with original Anfatec header included."""
    header_lines = [
        "WSxM file copyright UAM",
        "WSxM ASCII XYZ file",
        f"X[{unit_x}]\tY[{unit_y}]\tZ[{unit_z}]",
        "# --- Original Anfatec header ---"
    ] + ["# " + line.rstrip() for line in raw_txt_lines] + [
        "# --- End of original header ---"
    ]
    return "\n".join(header_lines)


# ---------------------------
# Channel processor
# ---------------------------

def process_channel(
    ch: Dict[str, str],
    base_path: str,
    mode: str,
    width: int,
    height: int,
    scan_x: float,
    scan_y: float,
    unit_x: str,
    unit_y: str,
    global_params: Dict[str, str],
    raw_txt_lines: List[str],
    basename: str,
    ts_str: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process one channel: filter, validate, and export.

    Returns
    -------
    export_entry : dict or None
        Metadata if valid and exported.
    invalid_entry : dict or None
        Metadata if invalid.
    """
    fname = ch["FileName"]
    caption = ch.get("Caption", fname)
    scale = float(ch["Scale"])
    unit_z = ch.get("PhysUnit", "a.u.")
    offset = float(ch.get("Offset", 0))

    # step 1: filter
    reason = filter_channel(ch, mode)
    if reason:
        return None, {
            "File": fname,
            "Channel": caption,
            "Status": "invalid",
            "Reason": reason,
            "Mode": mode,
            "ActiveFraction": "NA",
            "Cropped": "NA"
        }

    # step 2: validate
    img, status, active_frac_str = validate_channel(ch, base_path, width, height)
    if status != "valid" or img is None:
        return None, {
            "File": fname,
            "Channel": caption,
            "Status": "invalid",
            "Reason": status,
            "Mode": mode,
            "ActiveFraction": active_frac_str,
            "Cropped": "NA"
        }

    # step 3: export
    return (
        export_channel(
            fname, caption, img, scale, offset, width, height,
            scan_x, scan_y, unit_x, unit_y, unit_z,
            global_params, raw_txt_lines, basename, ts_str,
            active_frac_str, mode, base_path
        ),
        None
    )


# ---------------------------
# Main export function
# ---------------------------

def anfatec_to_wsxm(
    txt_file: str, base_path: str = ""
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert an Anfatec dataset (txt + bin) to WSxM .xyz files.

    Parameters
    ----------
    txt_file : str
        Path to Anfatec .txt file.
    base_path : str
        Directory containing the file and binaries.

    Returns
    -------
    exports : list of dict
        Metadata for valid exports.
    invalids : list of dict
        Metadata for invalid channels.
    """
    txt_path = os.path.join(base_path, txt_file)
    global_params, channels, raw_txt_lines = parse_anfatec_txt(txt_path)

    width = int(global_params["xPixel"])
    height = int(global_params["yPixel"])
    scan_x = float(global_params["XScanRange"])
    scan_y = float(global_params["YScanRange"])
    unit_x = global_params.get("XPhysUnit", "µm")
    unit_y = global_params.get("YPhysUnit", "µm")

    raw_date = global_params.get("Date", "")
    raw_time = global_params.get("Time", "")
    dt = parse_timestamp(raw_date, raw_time)
    ts_str = dt.strftime("%Y%m%d_%H%M%S") if dt else "unknowntime"

    basename = os.path.splitext(os.path.basename(txt_file))[0]
    mode = detect_mode(channels, base_path, width, height)

    exports, invalids = [], []
    for ch in channels:
        export_entry, invalid_entry = process_channel(
            ch, base_path, mode, width, height,
            scan_x, scan_y, unit_x, unit_y,
            global_params, raw_txt_lines, basename, ts_str
        )
        if export_entry:
            exports.append(export_entry)
        elif invalid_entry:
            invalids.append(invalid_entry)

    return exports, invalids
