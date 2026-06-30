import os
import sqlite3
import numpy as np
import scipy.io as sio


def initialize_local_database():
    """Initializes the SQLite database using your exact required schema."""
    connection = sqlite3.connect("local_wind_data.db")
    cursor = connection.cursor()

    print("🛠️ Re-initializing local database tables...")

    # Enable Foreign Key support
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop existing tables if rebuilding completely to avoid schema conflicts
    cursor.execute("DROP TABLE IF EXISTS taps;")
    cursor.execute("DROP TABLE IF EXISTS origin_models;")

    # 1. Create the Origin Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS origin_models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. Create the Taps Table with Mean and Std Dev columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS taps (
            tap_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            face_no INTEGER NOT NULL,
            x_coordinate REAL NOT NULL,
            y_coordinate REAL NOT NULL,
            mean_pressure REAL NOT NULL,
            std_dev_pressure REAL NOT NULL,
            FOREIGN KEY (model_id) REFERENCES origin_models(model_id) ON DELETE CASCADE
        );
    """)

    connection.commit()
    connection.close()
    print(
        "🚀 Local database structure updated successfully! Ready for statistical processing."
    )


def populate_database_from_mat(mat_file_path: str):
    """Loads a TPU .mat file, computes tap statistics, and populates the database."""
    if not os.path.exists(mat_file_path):
        print(f"❌ Error: File '{mat_file_path}' not found.")
        return

    print(f"📦 Extracting TPU data from: {mat_file_path}...")
    raw_data = sio.loadmat(mat_file_path)

    # Helper to resolve naming variations inside different TPU releases
    def _get_var(possible_names):
        for name in possible_names:
            for key in raw_data.keys():
                if key.lower().strip() == name.lower().strip():
                    return raw_data[key]
        return None

    cp_matrix = _get_var(["Wind_pressure_coefficients", "Cp"])
    loc_matrix = _get_var(["Location_of_measured_points", "Location"])

    if cp_matrix is None or loc_matrix is None:
        print("❌ Error: Missing core data matrices inside the .mat file.")
        return

    filename = os.path.basename(mat_file_path)
    num_points = loc_matrix.shape[1]

    # Connect to the database
    connection = sqlite3.connect("local_wind_data.db")
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    try:
        # 1. Insert the file origin metadata record
        cursor.execute(
            """
            INSERT INTO origin_models (filename) 
            VALUES (?) 
            ON CONFLICT(filename) DO NOTHING;
        """,
            (filename,),
        )

        # Retrieve the generated model_id for this specific file
        cursor.execute(
            "SELECT model_id FROM origin_models WHERE filename = ?;",
            (filename,),
        )
        model_id = cursor.fetchone()[0]

        print(
            f"🔄 Calculating stats and writing records for {num_points} pressure taps..."
        )

        taps_batch = []
        for i in range(num_points):
            # Extract spatial geometry data from the location matrix columns
            x_coord = float(loc_matrix[0, i])
            y_coord = float(loc_matrix[1, i])
            face_no = int(loc_matrix[3, i])

            # Isolate the corresponding column of time-series records to compute statistics
            time_series = cp_matrix[:, i]
            mean_pressure = float(np.mean(time_series))
            std_dev_pressure = float(np.std(time_series))

            # Store the record tuple for batch execution
            taps_batch.append(
                (
                    model_id,
                    face_no,
                    x_coord,
                    y_coord,
                    mean_pressure,
                    std_dev_pressure,
                )
            )

        # 2. Bulk insert all tap aggregates efficiently
        cursor.executemany(
            """
            INSERT INTO taps (model_id, face_no, x_coordinate, y_coordinate, mean_pressure, std_dev_pressure)
            VALUES (?, ?, ?, ?, ?, ?);
        """,
            taps_batch,
        )

        connection.commit()
        print(f"🎉 Database completely populated from '{filename}'!")

    except Exception as e:
        connection.rollback()
        print(f"❌ Database Transaction Failed: {e}")
    finally:
        connection.close()


# =====================================================================
# Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    # 1. Clean and initialize your schema tables
    initialize_local_database()

    # 2. Provide the file name of your downloaded TPU dataset file
    target_mat_file = r"C:\FINAL_SUMMER_PROJ\TPU_TEST_FILES\WIND_LOW_RISE_WITH_EAVE\Cp_ts_ROH12_deg000.mat"

    # 3. Parse and load stats directly into the tables
    populate_database_from_mat(target_mat_file)