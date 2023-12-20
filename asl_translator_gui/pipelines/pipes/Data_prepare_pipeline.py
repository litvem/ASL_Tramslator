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
from website.models import *
from data.upload_retraining_json_to_db import DataHandler


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


#takes frames from videos and then extracts key features of each frame and saves it in a np array and then save it under a video folder
def save_frames(file_name, sequence_length, path_to_np_video):
    #open the video using cv2
    video = cv2.VideoCapture(file_name)
    #height and width
    h, w = 0, 0

    #for the entire video
    while video.isOpened():
        #tools for exrtracting the keypoints
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            #only for the first 60 frames
            for frame_num in range(sequence_length):
                #read a frame
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

        #release the video and close cv2
        video.release()
        cv2.destroyAllWindows()

#downloads a video/datapoint
def save_videos(data_point, videos_address, sequence_length, np_address, video_id):
   
    #the value of the datapoint
    label = data_point["clean_text"]

    #make a directory for the video that will then hold some number of videos 
    dir_name = videos_address + "/" + label
    #name of the video and its path
    file_name = dir_name + "/" + "current.mp4"

    #data for cutting the downloaded video
    start_time = data_point["start_time"]
    end_time = data_point["end_time"]

    #the downloader
    ydl_opts = {
        'outtmpl': dir_name + "/" + "current" + ".%(ext)s",
        'format': 'bestvideo[ext=mp4]', "merge_output_format": 'mp4', 'ignoreerrors': True
    }

    #download video
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            #download the url 
            result = ydl.download(url_list=[data_point["url"]])
            if result == 0:
                print("Download successful!")
            else:
                print("Download failed. youtube_dl returned:", result)
                return False

        except youtube_dl.DownloadError as e:
            print("Error during download:", e)
    # crop video
    subprocess.run(['ffmpeg', '-y', '-i',
                     dir_name + "/" + "current" + ".mp4",
                     '-ss', str(start_time), '-t', str(end_time - start_time), file_name])

    # open video
    path_to_np_label = os.path.join(np_address, label)
    path_to_np_video = os.path.join(np_address, label, str(video_id))

    #make a directory for that video which will then hold some number of np arrays
    if not os.path.exists(path_to_np_label):
        os.mkdir(path_to_np_label)
        
    os.mkdir(path_to_np_video)

    #extracts the key features and then saves 60 frames per video
    save_frames(file_name, sequence_length, path_to_np_video)

    if os.path.exists(file_name):
        # Delete the file
        os.remove(file_name)
        print(f"The file '{file_name}' has been deleted.")
        return True
    else:
        print(f"The file '{file_name}' does not exist.")   


#gets the training data from the database
def prepare_data(X, DATA_PATH, actions, sequence_length, videos_folder, DB_path):
    
    last_uploaded_json_file = json.loads((Training_input.objects.latest('tr_input_id').tr_input_file).read().decode('utf-8'))
    print(Training_input.objects.latest('tr_input_id').tr_input_file)

    data_handler = DataHandler(db_file = os.path.abspath('data/data.db'))
    data_handler.insert_data(json_file=last_uploaded_json_file)
    clean_texts = [entry.get("clean_text", "") for entry in last_uploaded_json_file]
    actions_O = np.array(clean_texts)
    print(actions_O)
    actions = actions_O

    # Connect to the SQLite database
    with sqlite3.connect(DB_path) as connection:
        cursor = connection.cursor()

        #create a list of all our actions
        clean_text_values = actions.tolist()
        
        #query for reading all the data ****replace "MSASL_DATA" with the name of the table that holds the data
        query = 'SELECT * FROM RETRAINING_DATA_8 WHERE clean_text IN ({})'.format(','.join(['?'] * len(clean_text_values)))
        #executes the query 
        cursor.execute(query, clean_text_values)

        # Fetch all rows
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Convert the data to a list of dictionaries
        data_list = [dict(zip(columns, row)) for row in rows]

        # Convert the list of dictionaries to JSON format
        data_json = json.dumps(data_list)  # indent for pretty formatting
        #the data we end up using in json format
        parsed_data_json = json.loads(data_json)
        
        print(len(parsed_data_json))

    #create a folder for our data
    if not os.path.exists(os.path.abspath("data/MP_Data")):
        os.makedirs(os.path.abspath("ASL-translator/asl_translator_gui/data") + "MP_Data")
    #create a root directory for our videos
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)
    #create a root directory for all our np arrays
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    #create individual directories for our actions in our video folder
    for action in actions: 
        try: 
            os.makedirs(os.path.join(videos_folder, action))
        except:
            pass

    #used for naming the final np arrays
    video_id = 0
    
    print('here you two, this is the number of instances i found: ', len(parsed_data_json))

    #for all the actions, go through all our data points (25000) and if the data point is equal to the current action, do the logic
    for action in actions:
        for data_point in parsed_data_json:
            if data_point["clean_text"] == action:
                #download the video and return the result
                result = save_videos(data_point, videos_folder, sequence_length, DATA_PATH, video_id)
                #if successful, incrment the data id 
                if result is True:
                    video_id += 1
                
        print(video_id)

    
    table_name = 'RETRAINING_DATA_8'
    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
    
    connection.commit()
    try:
        cursor.execute(query, clean_text_values)
        print('hahah i could')
    except sqlite3.OperationalError as e:
        print("the table was deleted!", e)    
    
    connection.close()

# Path for exported data, numpy arrays
DATA_PATH_O = os.path.join(os.path.abspath("data/MP_Data")) 

# Actions that we try to detect
actions_O = np.array([
    'nice',
    'teacher',
    'no',
    'like',
    'want',
    'deaf',
    'hello',
    'I',
    'yes',
    'you',
    'pineapple',
    'father',
    'thank you',
    'beautiful',
    'fall'
                   ])

# Videos are going to be 60 frames in length
sequence_length_O = 60
#path were the actual mp4 videos get stored and then deleted
videos_folder_O = os.path.abspath("data/videos")
#path to the db were we hold the json to db formated data 
DB_path_O = os.path.abspath('data/data.db')
# DB_path_O = 'C:/Users/yasi7/Downloads/data/gui_new/ASL-translator/asl_translator_gui/data/data.db'
#create the pipeline given the argument
data_pipeline = Pipeline([
    ('prepare',FunctionTransformer(func=prepare_data,  
                                   kw_args={'DATA_PATH': DATA_PATH_O, 'actions': actions_O, 'sequence_length': sequence_length_O, 'videos_folder': videos_folder_O, 'DB_path': DB_path_O}))
])

#runs the pipeline ****ffor testing, comment later*****
# X_transformed = data_pipeline.fit_transform(None)
