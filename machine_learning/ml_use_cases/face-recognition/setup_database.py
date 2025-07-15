import sqlite3

def setup_database():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, timestamp TEXT)''')

    conn.commit()
    conn.close()

# Run the setup
if __name__ == '__main__':
    setup_database()
