import os
import random
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import json
from website.models import *
from data.upload_retraining_json_to_db import DataHandler
from sklearn.metrics import accuracy_score
import tensorflow as tf
import shutil


#path to our np arrays
DATA_PATH = os.path.join(os.path.abspath('data/MP_Data')) 
#list of all our labels
actions = np.array([
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
directory_path = os.path.abspath("media/input")


#class for the model trainer
class TrainModelTransformer(BaseEstimator, TransformerMixin):
    #constructor for the class, gets the number of epochs, number of frames to include and the list of lebels
    def __init__(self, actions,  target_file_count=30, epochs=55):
        self.target_file_count = target_file_count
        self.epochs = epochs
        self.target_file_count = target_file_count
        self.actions = actions
        self.model = None
        self.accuracy = 0
        self.train_accuracy = 0


    def fit(self, X, y=None):
        #for visulization and testing
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        #labels the actions 
        label_map = {label: num for num, label in enumerate(self.actions)}
        #lists for our final videos (np arrays) and labels
        sequences, labels = [], []
        #for all the actions and all their videos (folders of np arrays)
        
        last_uploaded_json_file = json.loads((Training_input.objects.latest('tr_input_id').tr_input_file).read().decode('utf-8'))
        clean_texts = [entry.get("clean_text", "") for entry in last_uploaded_json_file]
        actions_retrain= np.array(clean_texts)
        print(actions_retrain)
        
        self.actions = actions_retrain

        data_addresses = ["data/MP_Data", "data/MP_Data_original"]

        for address in data_addresses:
            for action in actions_retrain:
                videos = os.listdir(os.path.abspath(address) + "/" + "{}".format(action))
                if ".DS_Store" in videos:
                    videos.remove(".DS_Store")
                #pick 2 random videos for each action
                #random_videos = random.sample(videos, 1)
                #for all those videos
                for sequence in videos:
                    window = []
                    if sequence != ".DS_Store":
                        #count the number of np arrays (frames) this video has
                        number_of_f = os.listdir(os.path.abspath(address) + "/" + "{}".format(action) + "/" + sequence)
                        f_size = len(number_of_f)

                        #if the number of frames is smaller/equal 29, get all the 29 frames and add them to the windows list
                        if f_size < 27:
                            continue
                        elif f_size == self.target_file_count:
                            for frame_num in range(3, self.target_file_count):
                                res = np.load(
                                    os.path.join(os.path.join(os.path.abspath(address)), action, sequence, "{}.npy".format(frame_num)))
                                window.append(res)
                        #if the number of frames is 60, get every other frame 
                        elif f_size >= 60:
                            for frame_num in range(3, f_size - 4, 2):
                                res = np.load(
                                    os.path.join(os.path.join(os.path.abspath(address)), action, sequence, "{}.npy".format(frame_num)))
                                window.append(res)
                        #if the number of frames is between 29 and 60, pick 30 frames out of all available
                        else:
                            random_numbers = random.sample(range(3, f_size), self.target_file_count)
                            random_numbers.sort()
                            for frame_num in random_numbers:
                                res = np.load(
                                    os.path.join(os.path.join(os.path.abspath(address)), action, sequence, "{}.npy".format(frame_num)))    
                                window.append(res)

                        sequences.append(window)
                        labels.append(label_map[action])

        
        #make a np array of all the sequences
        Z = np.array(sequences)
        #make the labels catagorical
        y = to_categorical(labels).astype(int)
        #split the data into train and test set with 90-10% ratio
        X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.1, stratify=y)
        
        print(X_train.shape)

        #model's architecture
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(27,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='tanh'))
        self.model.add(LSTM(64, return_sequences=False, activation='tanh'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        #compile the model and fit
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.model.fit(X_train, y_train, epochs=self.epochs, callbacks=[tb_callback])
        

        yhat = self.model.predict(X_test)

        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        
        # Evaluate the model on the train set
        _, self.train_accuracy = self.model.evaluate(X_train, y_train)

        # Evaluate the model on the test set
        self.accuracy = accuracy_score(ytrue, yhat)

    
        print("Model Accuracy on test set:", self.accuracy)

        print("Model Accuracy on train set:", self.train_accuracy)
        return self

    def transform(self, X):
        #return the results of the pipeline
        path_to_MP_Data = os.path.join(os.path.abspath("data/MP_Data"))
        actions_in_MP_Data = os.listdir(path_to_MP_Data)
        for action in actions_in_MP_Data:
            try:
                shutil.rmtree(os.path.join(path_to_MP_Data, action))
                print(f"Deleted: {action}")
            except Exception as e:
                print(f"Error deleting {action}: {e}")


        path_to_videos = os.path.join(os.path.abspath("data/videos"))
        actions_in_videos = os.listdir(path_to_videos)
        for action in actions_in_videos:
            try:
                shutil.rmtree(os.path.join(path_to_videos, action))
                print(f"Deleted: {action}")
            except Exception as e:
                print(f"Error deleting {action}: {e}")

        return {'model': self.model, 'accuracy': self.accuracy, 'train_accuracy': self.train_accuracy}



train_pipeline = Pipeline([
    ('preprocessAndFit', TrainModelTransformer(actions=actions,target_file_count=30, epochs=55))
])