import os
import mne
import pandas as pd

# Helper functions
def unpack_edf(edf_path):
        
    base = os.path.basename(edf_path)  
    name, _ = os.path.splitext(base)
    out_csv = "./" + f"{name}_csv.csv"

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')

    ch_names = raw.ch_names
    data     = raw.get_data()              
    times    = raw.times                
    
    df = pd.DataFrame(data.T, columns=ch_names)
    df.insert(0, 'time_s', times)
        
    df.to_csv(out_csv, index=False)
    print(f"Saved {df.shape[0]} samples × {df.shape[1]} channels to '{out_csv}'")
    return df

def remove_col(file, col_name):
    df = pd.read_csv(file)
    df = df.drop(col_name, axis=1)
    df.to_csv(file, index=False)