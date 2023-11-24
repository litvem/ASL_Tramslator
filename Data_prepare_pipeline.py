import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import re
import json
import yt_dlp as youtube_dl
import subprocess
import sqlite3
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


mp_holistic = mp.solutions.holistic # Holistic model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    result = np.concatenate([pose, face, lh, rh])
    return result

def save_frames(file_name, sequence_length, path_to_np_video):
    video = cv2.VideoCapture(file_name)
    h, w = 0, 0

    while video.isOpened():
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for frame_num in range(sequence_length):
                ret, frame = video.read()
                if not ret:
                    break

                ##extract features
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(path_to_np_video, str(frame_num))
                np.save(npy_path, keypoints)
               
        video.release()
        cv2.destroyAllWindows()

def save_videos(data_point, videos_address, sequence_length, np_address, video_id):
   
    label = data_point["clean_text"]
    
    dir_name = videos_address + "/" + label
    file_name = dir_name + "/" + "current.mp4"

    start_time = data_point["start_time"]
    end_time = data_point["end_time"]

    ydl_opts = {
        'outtmpl': dir_name + "/" + "current" + ".%(ext)s",
        'format': 'bestvideo[ext=mp4]', "merge_output_format": 'mp4', 'ignoreerrors': True
    }

    # download video

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.download(url_list=[data_point["url"]])
            if result is 0:
                print("Download successful!")
            else:
                print("Download failed. youtube_dl returned:", result)
                return False

        except youtube_dl.DownloadError as e:
            print("Error during download:", e)
    # crop video
    subprocess.call(['C:/Users/yasi7/anaconda3/pkgs/ffmpeg-4.3.1-ha925a31_0/Library/bin/ffmpeg.exe', '-y', '-i',
                     dir_name + "/" + "current" + ".mp4",
                     '-ss', str(start_time), '-t', str(end_time - start_time), file_name])

    # open video
    path_to_np_label = os.path.join(np_address, label)
    path_to_np_video = os.path.join(np_address, label, str(video_id))

    if not os.path.exists(path_to_np_label):
        os.mkdir(path_to_np_label)
        
    os.mkdir(path_to_np_video)

    save_frames(file_name, sequence_length, path_to_np_video)

    if os.path.exists(file_name):
        # Delete the file
        os.remove(file_name)
        print(f"The file '{file_name}' has been deleted.")
        return True
    else:
        print(f"The file '{file_name}' does not exist.")   


def prepare_data(X, DATA_PATH, actions, sequence_length, videos_folder, DB_path):
    # Connect to the SQLite database
    with sqlite3.connect(DB_path) as connection:
        cursor = connection.cursor()

        clean_text_values = actions.tolist()
        
        query = 'SELECT * FROM MSASL_DATA WHERE clean_text IN ({})'.format(','.join(['?'] * len(clean_text_values)))
        cursor.execute(query, clean_text_values)

        # Fetch all rows
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Convert the data to a list of dictionaries
        data_list = [dict(zip(columns, row)) for row in rows]

        # Convert the list of dictionaries to JSON format
        data_json = json.dumps(data_list)  # indent for pretty formatting
        
        parsed_data_json = json.loads(data_json)
        
        print(len(parsed_data_json))


    if not os.path.exists("test1"):
        os.makedirs("test1")
    
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    for action in actions: 
        try: 
            os.makedirs(os.path.join(videos_folder, action))
        except:
            pass

    video_id = 0

    for data_point in parsed_data_json:
        if data_point["clean_text"] in actions:
            result = save_videos(data_point, videos_folder, sequence_length, DATA_PATH, video_id)
            if result is True:
                video_id += 1
            
    print(video_id)



# Path for exported data, numpy arrays
DATA_PATH_O = os.path.join('test1/MP_Data') 
# Actions that we try to detect
actions_O = np.array(['hello','nice','meet','you'])
# Videos are going to be 30 frames in length
sequence_length_O = 60
videos_folder_O = "test1/videos"
DB_path_O = 'MS-ASL/data.db'

pipeline = Pipeline([
    ('prepare',FunctionTransformer(func=prepare_data,  
                                   kw_args={'DATA_PATH': DATA_PATH_O, 'actions': actions_O, 'sequence_length': sequence_length_O, 'videos_folder': videos_folder_O, 'DB_path': DB_path_O}))
])

X_transformed = pipeline.fit_transform(None)
