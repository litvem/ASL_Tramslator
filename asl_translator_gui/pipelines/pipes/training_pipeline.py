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
from joblib import dump, load
import json
from website.models import *
from data.upload_retraining_json_to_db import DataHandler
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import tensorflow as tf



#path to our np arrays
DATA_PATH = os.path.join(os.path.abspath('data/MP_Data')) 
DATA_PATH_TEST = os.path.join(os.path.abspath('data/test_set')) 
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

# path_to_model_weights = None
# Last_deployed_model = Training.objects.get(is_deployed = True)
# path_to_model_weights = Last_deployed_model.model_weights
# print(path_to_model_weights)

# json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
# if json_files:
#     # Get the full paths of the JSON files
#     json_paths = [os.path.join(directory_path, f) for f in json_files]

#     # Find the newest JSON file based on modification time
#     newest_json_path = max(json_paths, key=os.path.getmtime)
# last_uploaded_json_file = json.loads((Training_input.objects.latest('id').tr_input_file).read().decode('utf-8'))
# clean_texts = [entry.get("clean_text", "") for entry in last_uploaded_json_file]
# actions= np.array(clean_texts)
# print(actions)


#class for the model trainer
class TrainModelTransformer(BaseEstimator, TransformerMixin):
    #constructor for the class, gets the number of epochs, number of frames to include and the list of lebels
    def __init__(self, actions,  target_file_count=30, epochs=1000):
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
        
        for action in actions_retrain:
            videos = os.listdir(os.path.abspath("data/MP_Data") + "/" + "{}".format(action))
            if ".DS_Store" in videos:
                videos.remove(".DS_Store")
            #pick 2 random videos for each action
            random_videos = random.sample(videos, 1)
            #for all those videos
            for sequence in random_videos:
                window = []
                if sequence != ".DS_Store":
                    #count the number of np arrays (frames) this video has
                    number_of_f = os.listdir(os.path.abspath("data/MP_Data") + "/" + "{}".format(action) + "/" + sequence)
                    f_size = len(number_of_f)

                    #if the number of frames is smaller/equal 29, get all the 29 frames and add them to the windows list
                    if f_size < 27:
                        continue
                    elif f_size == self.target_file_count:
                        for frame_num in range(3, self.target_file_count):
                            res = np.load(
                                os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                            window.append(res)
                    #if the number of frames is 60, get every other frame 
                    elif f_size >= 60:
                        for frame_num in range(3, f_size - 4, 2):
                            res = np.load(
                                os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                            window.append(res)
                    #if the number of frames is between 29 and 60, pick 30 frames out of all available
                    else:
                        random_numbers = random.sample(range(3, f_size), self.target_file_count)
                        random_numbers.sort()
                        for frame_num in random_numbers:
                            res = np.load(
                                os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))    
                            window.append(res)

                    sequences.append(window)
                    labels.append(label_map[action])

        


        sequences_test, labels_test = [], []
        for action_test in self.actions:
            videos = os.listdir(os.path.abspath("data/test_set") + "/" + "{}".format(action_test))
            if ".DS_Store" in videos:
                videos.remove(".DS_Store")
            for sequence in videos:
                window = []
                if sequence != ".DS_Store":
                    #count the number of np arrays (frames) this video has
                    number_of_f = os.listdir(os.path.abspath("data/test_set") + "/" + "{}".format(action_test) + "/" + sequence)
                    f_size = len(number_of_f)

                    if f_size == self.target_file_count:
                        for frame_num in range(3, self.target_file_count):
                            res = np.load(
                                os.path.join(DATA_PATH_TEST, action_test, sequence, "{}.npy".format(frame_num)))
                            window.append(res)
                    
                    sequences_test.append(window)
                    labels_test.append(label_map[action_test])

    
        print(len(labels_test))
        #make a np array of all the sequences
        Z = np.array(sequences)
        #make the labels catagorical
        y = to_categorical(labels).astype(int)
        #split the data into train and test set with 80-20% ratio
        X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.2)

        K = np.array(sequences_test)
        P = to_categorical(labels_test).astype(int)
        _, X_test_set, _, y_test_set = train_test_split(K, P, test_size=0.99)
        print(X_test_set.shape)
        print(y_test_set.shape)
        print(label_map)
        
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
        
        
        Last_deployed_model = Training.objects.get(is_deployed = True)
        path_to_model_weights = str(Last_deployed_model.model_weights)
        print(path_to_model_weights)
        abs_path_to_model_weights = os.path.abspath("media/"+ path_to_model_weights)
        print(abs_path_to_model_weights)
        # path_to_model_weights = "C:/Users/yasi7/Downloads/data/gui_new/ASL-translator/asl_translator_gui/media/models/actions_solution_20l_v0_1.h5"
        
        #self.model.load_weights(abs_path_to_model_weights)
        self.model = tf.keras.models.load_model(os.path.abspath("media/models/"+ "MyModel_tf"))

        _, goz = self.model.evaluate(X_test_set, y_test_set)
        print('goz: ', goz)

        self.model.fit(X_train, y_train, epochs=self.epochs, callbacks=[tb_callback])
        

        yhat = self.model.predict(X_test_set)

        ytrue = np.argmax(y_test_set, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        
        # Evaluate the model on the train set
        _, self.train_accuracy = self.model.evaluate(X_train, y_train)

        # Evaluate the model on the test set
        self.accuracy = accuracy_score(ytrue, yhat)
        #self.accuracy = self.train_accuracy - 0.05
        print("Model Accuracy on test set:", self.accuracy)

        print("Model Accuracy on train set:", self.train_accuracy)
        return self

    def transform(self, X):
        #return the results of the pipeline
        path_to_MP_Data = os.path.join(os.path.abspath("data/MP_Data"))
        path_to_videos = os.path.join(os.path.abspath("data/videos"))
        files_in_MP_Data = os.listdir(path_to_MP_Data)
        files_in_videos = os.listdir(path_to_videos)
        for file_name in files_in_MP_Data:
            file_path = os.path.join(path_to_MP_Data, file_name)
            try:
                os.rmdir(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        for file_name in files_in_videos:
            file_path = os.path.join(path_to_MP_Data, file_name)
            try:
                os.rmdir(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        return {'model': self.model, 'accuracy': self.accuracy, 'train_accuracy': self.train_accuracy}



train_pipeline = Pipeline([
    ('preprocessAndFit', TrainModelTransformer(actions=actions,target_file_count=30, epochs=1000))
])

#runs the pipeline ****for testing, comment later***** 
# result = train_pipeline.fit_transform(None)
# trained_model = result['model']
# accuracy = result['accuracy']
# train_accuracy = result['train_accuracy']

# #save the model
# dump(trained_model, 'asl_translator_gui/trained_models/{}.joblib'.format(random.randint(1000, 9999)))
