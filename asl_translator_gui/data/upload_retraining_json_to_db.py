import sqlite3
import json
import os

class DataHandler:
    def __init__(self, db_file):
        # Connect to the SQLite database
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()

        # Create the table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS RETRAINING_DATA_5 (
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
        # with open(json_file, 'r', encoding='utf-8') as file:
        #     data = json.load(file)


        

    def insert_data(self, json_file):
        for row in json_file:
            box_json = json.dumps(row["box"])
            # Assuming the keys in the JSON file match the table columns
            self.cursor.execute('''INSERT INTO RETRAINING_DATA_5 (org_text, clean_text, start_time, signer_id, signer, start, end, file, label, height, fps, end_time, url, text, box, width) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                            (row["org_text"], row["clean_text"], row["start_time"], row["signer_id"], 
                            row["signer"], row["start"], row["end"], row["file"], 
                            row["label"], row["height"], row["fps"], row["end_time"], 
                            row["url"], row["text"], box_json, row["width"]))
        print('i added all them motherfuckers')
        self.connection.commit()
        self.connection.close()

