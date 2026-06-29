import sqlite3

def initialize_local_database():
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
    print("🚀 Local database structure updated successfully! Ready for statistical processing.")

if __name__ == "__main__":
    initialize_local_database()