"""
STAGE 1: Data Loading
Load LVM files and save to RAW_DATA.pkl
"""

import os
import re
import pandas as pd
import pickle
from config import *

def load_lvm_file(filepath):
    """Load single LVM file"""
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    data_start = None
    header_count = 0
    for i, line in enumerate(lines):
        if '***End_of_Header***' in line:
            header_count += 1
            if header_count == 2:
                data_start = i + 2
                break
    
    if data_start is None:
        return None
    
    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if line and not line.startswith('X_Value'):
            parts = line.replace(',', '.').split('\t')
            if len(parts) >= 4:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
                except:
                    pass
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=['X_Value', 'Impac', 'Keller', 'TC'])
    max_idx = df['TC'].idxmax()
    df = df.iloc[max_idx:].reset_index(drop=True)
    df = df[(df['TC'] >= MIN_TEMP) & (df['TC'] <= MAX_TEMP)].reset_index(drop=True)
    
    if len(df) == 0:
        return None
    
    df['Time'] = df['X_Value'] - df['X_Value'].iloc[0]
    return df


def load_all_data():
    """Load all LVM files from all directories"""
    print("\n" + "="*70)
    print("STAGE 1: LOADING LVM FILES")
    print("="*70)
    
    all_data = []
    soaking_map = {'SHORT': 0, 'STANDARD': 1, 'LONG': 2}
    
    for soaking_type, directory in DATA_DIRS.items():
        print(f"\n📂 {soaking_type}: {directory}")
        
        if not os.path.exists(directory):
            print(f"  ⚠️  Directory not found")
            continue
        
        files = [f for f in os.listdir(directory) if f.lower().endswith('.lvm')]
        print(f"  Found {len(files)} LVM files")
        
        for filename in sorted(files):
            try:
                filepath = os.path.join(directory, filename)
                df = load_lvm_file(filepath)
                
                if df is not None and len(df) > 20:
                    match = re.match(r'(\d+)mm_test_(\d+)\.lvm', filename, re.IGNORECASE)
                    thickness_mm = int(match.group(1)) if match else 0
                    
                    df['soaking_encoded'] = soaking_map[soaking_type]
                    df['soaking_label'] = soaking_type
                    df['thickness_mm'] = thickness_mm
                    df['filename'] = filename
                    
                    all_data.append(df)
                    print(f"    ✓ {filename}: {len(df)} points")
            except Exception as e:
                print(f"    ✗ {filename}: {e}")
    
    if not all_data:
        print("\n❌ No data loaded!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total samples: {len(combined_df):,}")
    print(f"   Columns: {combined_df.shape[1]}")
    
    # Save to PKL
    with open(RAW_DATA_PKL, 'wb') as f:
        pickle.dump(combined_df, f)
    print(f"\n💾 Saved: {RAW_DATA_PKL}")
    
    return combined_df


if __name__ == "__main__":
    load_all_data()