import csv

def write_manifest(records, out_file):
    if not records:
        return
    keys = records[0].keys()
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
