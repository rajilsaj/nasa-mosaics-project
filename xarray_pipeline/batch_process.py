from pathlib import Path
from glob import glob
import os
from pds4_to_xarray import process_file

input_dir = './data'
output_dir = './data/processed'

# Recursively find XML files in subfolders
print("Finding XML files...")
xml_files = glob(os.path.join(input_dir, '**', '*.xml'), recursive=True)
print(f"Found {len(xml_files)} XML files.")

for xml_file in xml_files:
    rel_path = os.path.relpath(xml_file, input_dir)
    rel_nc_path = os.path.splitext(rel_path)[0] + '.nc'
    output_path = os.path.join(output_dir, rel_nc_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Skipping {xml_file}: already exists")
        continue

    try:
        print(f"Processing {xml_file}")
        ds = process_file(xml_path=xml_file)
        ds.to_netcdf(output_path)
        print(f"   ✅ Saved to {output_path}")
    except Exception as e:
        print(f"❌ ERROR processing {xml_file}: {e}")
