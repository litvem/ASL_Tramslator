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

#path to our np arrays
DATA_PATH = os.path.join('asl_translator_gui/data/MP_Data') 
#list of all our labels
actions = ['nice','teacher','eat','no','happy','like','orange','want','deaf','school','sister','finish','white',
                      'what','tired','friend','sit','yes','student','spring','good','hello','mother','fish','again','learn',
                      'sad','table','where','father','milk','paper','forget','cousin','brother','nothing','book','girl','fine',
                      'black']


#class for the model trainer
class TrainModelTransformer(BaseEstimator, TransformerMixin):
    #constructor for the class, gets the number of epochs, number of frames to include and the list of lebels
    def __init__(self, actions=actions,  target_file_count=29, epochs=10):
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
        for action in self.actions:
            videos = os.listdir("asl_translator_gui/data/MP_Data" + "/" + "{}".format(action))
            if ".DS_Store" in videos:
                videos.remove(".DS_Store")
            #pick 38 random videos for each action
            random_videos = random.sample(videos, 38)

            #for all those videos
            for sequence in random_videos:
                window = []
                if sequence != ".DS_Store":
                    #count the number of np arrays (frames) this video has
                    number_of_f = os.listdir("asl_translator_gui/data/MP_Data" + "/" + "{}".format(action) + "/" + sequence)
                    f_size = len(number_of_f)

                    #if the number of frames is smaller/equal 29, get all the 29 frames and add them to the windows list
                    if f_size <= self.target_file_count:
                        for frame_num in range(self.target_file_count):
                            res = np.load(
                                os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                            window.append(res)
                    #if the number of frames is 60, get every other frame 
                    elif f_size >= 60:
                        for frame_num in range(0, f_size - 2, 2):
                            res = np.load(
                                os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                            window.append(res)
                    #if the number of frames is between 29 and 60, pick 30 frames out of all available
                    else:
                        random_numbers = random.sample(range(0, 30), self.target_file_count)
                        random_numbers.sort()
                        for frame_num in random_numbers:
                            res = np.load(
                                os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))    
                            window.append(res)

                    sequences.append(window)
                    labels.append(label_map[action])
    

        #make a np array of all the sequences
        Z = np.array(sequences)
        #make the labels catagorical
        y = to_categorical(labels).astype(int)
        #split the data into train and test set with 80-20% ratio
        X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.2)

        #model's architecture
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(29,1662)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128, return_sequences=True, activation='tanh'))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, return_sequences=False, activation='tanh'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(np.array(actions).shape[0], activation='softmax'))

        #compile the model and fit
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.fit(X_train, y_train, epochs=self.epochs, callbacks=[tb_callback])
        
        # Evaluate the model on the test set
        _, self.accuracy = self.model.evaluate(X_test, y_test)
        print("Model Accuracy on test set:", self.accuracy)

        # Evaluate the model on the train set
        _, self.train_accuracy = self.model.evaluate(X_train, y_train)
        print("Model Accuracy on train set:", self.train_accuracy)
        return self

    def transform(self, X):
        #return the results of the pipeline
        return {'model': self.model, 'accuracy': self.accuracy, 'train_accuracy': self.train_accuracy}


train_pipeline = Pipeline([
    ('preprocessAndFit', TrainModelTransformer(actions=actions,target_file_count=29, epochs=5000))
])

#runs the pipeline ****for testing, comment later***** 
result = train_pipeline.fit_transform(None)
trained_model = result['model']
accuracy = result['accuracy']
train_accuracy = result['train_accuracy']

#save the model
dump(trained_model, 'asl_translator_gui/trained_models/{}.joblib'.format(random.randint(1000, 9999)))
