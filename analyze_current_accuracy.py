import sqlite3
import pandas as pd
import os

# Connect to the database
db_path = 'C:\\Dev\\jsat3d\\jsats3d\\output\\jsats3d_2025_tagdrag_20250610_anchorZOI03.db'

if os.path.exists(db_path):
    print(f"Database exists: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Examine key aspects that relate to the accuracy problems
        
        print("\n=== RECEIVER COORDINATES ANALYSIS ===")
        # Look at receiver coordinates
        receiver_df = pd.read_sql('SELECT Rec_ID, X_t, Y_t, Z_t FROM tblReceiver ORDER BY Rec_ID', conn)
        print("Receiver coordinates (UTM Easting/Northing/Depth):")
        print(receiver_df)
        
        print("\n=== RECEIVER Z COORDINATES (DEPTHS) ===")
        print("Z coordinates (depth below surface, negative values):")
        z_coords = receiver_df[['Rec_ID', 'Z_t']].sort_values('Z_t')
        print(z_coords)
        
        print("\n=== POSITIONING RESULTS ANALYSIS ===")  
        # Examine what position data we have
        pos_cols = pd.read_sql('PRAGMA table_info(tblPositions_Deng)', conn)
        print("Columns in tblPositions_Deng:")
        for col in pos_cols['name']:
            print(f"  {col}")
            
        # Get some position data to see the structure
        pos_data = pd.read_sql('SELECT Tag_ID, transNo, X, Y, Z, solution, comment FROM tblPositions_Deng LIMIT 10', conn)
        print("\nSample position data:")
        print(pos_data)
        
        print("\n=== MULTIPATH FILTERING ANALYSIS ===")
        # Look at fish multipath filtering results
        fish_filter = pd.read_sql('''
            SELECT Rec_ID, multipath_prediction, COUNT(*) as count 
            FROM tblDetectionFilterSecondary 
            WHERE Tag_ID IN ('FC36', 'FFD3')
            GROUP BY Rec_ID, multipath_prediction 
            ORDER BY Rec_ID, multipath_prediction
        ''', conn)
        print("Fish multipath filtering results (FC36/FFD3 only):")
        print(fish_filter)
        
        print("\n=== ACCURACY METRICS FROM JOURNAL ===")
        print("From the journal, we know:")
        print("- Accuracy with 9 receivers (box rule): FC36 7.0m/5.3m median error, FFD3 5.3m/4.7m")
        print("- With swap, accuracy improved to: FC36 5.4m/4.7m, FFD3 4.7m/5.2m") 
        print("- The swap lifted yield from 0.63/0.47 to 0.80/0.66")
        print("- Remaining issues: ZOI01 position (7m off), ZOI02 instability, flat array")
        
        print("\n=== PROBLEM IDENTIFICATION ===")
        print("Based on journal information, the key problems are:")
        print("1. ZOI05 and ZOI06 positions were incorrect - causing large residuals")
        print("2. ZOI01 has un-constrained Z coordinate (depth) - 7m error in x") 
        print("3. ZOI02 has unstable clock (hundreds of steps over 1ms, up to 323ms)")
        print("4. Flat array geometry makes Z-axis poorly constrained")
        print("5. The 'what-if' swap fixed the position errors but accuracy still not sub-meter")
        
        conn.close()
        
    except Exception as e:
        print(f"Error examining database: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Database does not exist: {db_path}")