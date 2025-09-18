import os
import csv
from tqdm import tqdm
from anfatec_parser import anfatec_to_wsxm

def write_manifest(file, records, overwrite=False):
    """Write CSV manifest file with overwrite or append."""
    if not records:
        return
    mode = "w" if overwrite else "a"
    header = list(records[0].keys())
    with open(file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if overwrite:
            writer.writeheader()
        writer.writerows(records)

def batch_export(base_path):
    files = [f for f in os.listdir(base_path) if f.endswith(".txt")]
    print(f"Found {len(files)} .txt files to process\n")

    valid_manifest = os.path.join(base_path, "valid_files_log.csv")
    invalid_manifest = os.path.join(base_path, "invalid_files_log.csv")

    # Always overwrite manifests at the start
    if os.path.exists(valid_manifest):
        os.remove(valid_manifest)
    if os.path.exists(invalid_manifest):
        os.remove(invalid_manifest)

    for fname in tqdm(files, desc="Exporting", unit="file"):
        try:
            exports, invalids = anfatec_to_wsxm(fname, base_path=base_path)

            if exports:
                write_manifest(valid_manifest, exports, overwrite=not os.path.exists(valid_manifest))
            if invalids:
                write_manifest(invalid_manifest, invalids, overwrite=not os.path.exists(invalid_manifest))

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "data")
    batch_export(base_path)
