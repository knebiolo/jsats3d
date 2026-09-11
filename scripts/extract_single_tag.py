"""Extract a single tag's detections from the master detection CSV."""
import pandas as pd

# --- settings ---
input_csv = r"K:\Jobs\5662\001\Analysis\Data\CowlitzAT2025_Data_Deliverables\2_AT_detection_datasets\master_df_test.csv"
output_csv = r"K:\Jobs\5662\001\Analysis\Data\CowlitzAT2025_Data_Deliverables\2_AT_detection_datasets\single_tag_df_test.csv"
tag_code = "FFD3"   # the one tag to pull out

# --- load, filter, save ---
df = pd.read_csv(input_csv)
tag_df = df[df.tagCode == tag_code]
tag_df.to_csv(output_csv, index=False)

print(f"Tag {tag_code}: {len(tag_df)} of {len(df)} rows written to {output_csv}")
