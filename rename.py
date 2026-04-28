"""
Rename PDFs from sequential numbers to EID-based names
Run this first before the extraction script
"""

import os
import pandas as pd

# Configuration
CSV_PATH = 'srl-full-txt-review.csv'  # Your CSV file
PDF_FOLDER = 'pdf-files'               # Folder containing numbered PDFs
EID_COLUMN = 'DocumentID'                     # Column with Scopus EIDs

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records from CSV")

# Check if PDF folder exists
if not os.path.exists(PDF_FOLDER):
    print(f"Error: PDF folder '{PDF_FOLDER}' not found!")
    exit()

# Create backup folder
backup_dir = os.path.join(PDF_FOLDER, 'backup_original')
os.makedirs(backup_dir, exist_ok=True)

# Rename PDFs
renamed = 0
skipped = 0
errors = []

for i, eid in enumerate(df[EID_COLUMN].head(239), 1):  # Adjust 239 to your total
    old_name = f"{i}.pdf"
    old_path = os.path.join(PDF_FOLDER, old_name)
    
    # Clean EID for filename (remove characters that might cause issues)
    safe_eid = str(eid).replace('/', '_').replace('\\', '_').replace(':', '_')
    new_name = f"{safe_eid}.pdf"
    new_path = os.path.join(PDF_FOLDER, new_name)
    
    if os.path.exists(old_path):
        try:
            os.rename(old_path, new_path)
            renamed += 1
            if i <= 5:  # Show first 5 for verification
                print(f"✓ Renamed: {old_name} → {new_name}")
        except Exception as e:
            errors.append(f"Error renaming {old_name}: {e}")
            skipped += 1
    else:
        skipped += 1
        if i <= 10:  # Show first few missing files
            print(f"⚠ Not found: {old_path}")

print(f"\n{'='*50}")
print(f"Renamed: {renamed}")
print(f"Skipped/Missing: {skipped}")
if errors:
    print(f"Errors: {len(errors)}")
    for err in errors[:5]:
        print(f"  {err}")