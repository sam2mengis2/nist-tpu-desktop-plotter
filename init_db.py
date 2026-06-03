import psycopg2

# Paste your copied connection string (URI) from Supabase here
# It should start with "postgresql://"
DB_PASSWORD = "$Web4now$03"

def initialize_database():
    commands = (
        """
        CREATE TABLE IF NOT EXISTS buildings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name TEXT NOT NULL UNIQUE,
            scale_factor FLOAT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS taps (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
            tap_label TEXT NOT NULL,
            x_coord FLOAT NOT NULL,
            y_coord FLOAT NOT NULL,
            z_coord FLOAT NOT NULL,
            UNIQUE (building_id, tap_label)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_taps_building ON taps(building_id);
        """
    )
    
    conn = None
    try:
        print("Connecting to Supabase...")
        
        # Connect using explicit parameters to bypass URI parsing errors
        conn = psycopg2.connect(
            host="aws-1-us-east-1.pooler.supabase.com",
            port="5432",
            database="postgres",
            user="postgres.nanddzdspaucmwlyoyoc",
            password=DB_PASSWORD
        )
        
        cur = conn.cursor()
        
        print("Creating tables...")
        for command in commands:
            cur.execute(command)
            
        cur.close()
        conn.commit()
        print("Success! Tables created in Supabase.")
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    initialize_database()