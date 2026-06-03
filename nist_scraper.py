from pyhdf.SD import SD, SDC
import os

def explore_hdf4(file_path):
    print(f"--- Scanning {os.path.basename(file_path)} ---")
    try:
        hdf = SD(file_path, SDC.READ)
        datasets_dict = hdf.datasets()
        
        print("\nFound the following datasets:")
        for idx, (name, info) in enumerate(datasets_dict.items()):
            # info is a tuple containing metadata like dimensions, data type, etc.
            dimensions = info[0]
            shape = info[1]
            data_type = info[2]
            print(f"[{idx}] Name: '{name}' | Shape: {shape} | Type: {data_type}")
            
    except Exception as e:
        print(f"Failed to open HDF4: {e}")
    finally:
        try:
            hdf.end()
        except:
            pass

# Point this to one of your unzipped HDF files!
TARGET_FILE = r"c:\Users\sam2m\Downloads\m11\ADW300o100D016a3000\ADW300o100D016a3000.HDF" # Adjust path if needed
explore_hdf4(TARGET_FILE)