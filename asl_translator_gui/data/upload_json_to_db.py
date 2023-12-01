import sqlite3
import json

# Connect to the SQLite database
connection = sqlite3.connect('your_path')
cursor = connection.cursor()

# Create the table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS MSASL_DATA (
        org_text TEXT,
        clean_text TEXT,
        start_time REAL,
        signer_id INTEGER,
        signer INTEGER,
        start INTEGER,
        end INTEGER,
        file TEXT,
        label INTEGER,
        height REAL,
        fps REAL,
        end_time REAL,
        url TEXT,
        text TEXT,
        box TEXT,
        width REAL
    )
''')

# Read JSON data and insert into the database
with open('MS-ASL/MSASL_data.json', 'r', encoding='utf-8') as json_file:

    data = json.load(json_file)

    for row in data:
        cur_str = json.dumps(row)                        
        cur_row = json.loads(cur_str)

        box_json = json.dumps(cur_row["box"])
        # Assuming the keys in the JSON file match the table columns
        cursor.execute('''INSERT INTO MSASL_DATA (org_text, clean_text, start_time, signer_id, signer, start, end, file, label, height, fps, end_time, url, text, box, width) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                        (cur_row["org_text"], cur_row["clean_text"], cur_row["start_time"], cur_row["signer_id"], 
                        cur_row["signer"], cur_row["start"], cur_row["end"], cur_row["file"], 
                        cur_row["label"], cur_row["height"], cur_row["fps"], cur_row["end_time"], 
                        cur_row["url"], cur_row["text"], box_json, cur_row["width"]))

# Commit changes and close the connection
connection.commit()
connection.close()
