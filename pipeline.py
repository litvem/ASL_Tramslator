import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import FunctionTransformer


def load_data(actions, target_file_count=29):
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        videos = os.listdir(DATA_PATH+ "{}".format(action))
        if ".DS_Store" in videos:
            videos.remove(".DS_Store")
        
        random_videos = random.sample(videos, 33)

        for sequence in random_videos:
            window = []
            if sequence != ".DS_Store":
                number_of_f = os.listdir(DATA_PATH + "{}".format(action) + "/" + sequence)
                f_size = len(number_of_f)

                if f_size <= target_file_count:
                    for frame_num in range(target_file_count):
                        res = np.load(os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                        window.append(res)
                elif f_size >= 60:
                    for frame_num in range(0, f_size-2, 2):
                        res = np.load(os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                        window.append(res)
                else:
                    random_numbers = random.sample(range(0, 29 + 1), target_file_count)
                    random_numbers.sort()
                    for frame_num in random_numbers:
                        res = np.load(os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                        window.append(res)
                
                sequences.append(window)
                labels.append(label_map[action])

    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(sequences, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test


        
actions = ["hello", "nice", "meet", "you"]        
DATA_PATH = "./MP_Data"  
X_train, X_test, y_train, y_test = load_data(actions)


def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(29, 1662)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def evaluate_model(model, X_test, y_test, actions):
    y_pred = model.predict(X_test)
    
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    
    class_report = classification_report(y_true, y_pred, target_names=actions)
    print('Classification Report:\n', class_report)

    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    print('Confusion Matrices:')
    for i, matrix in enumerate(confusion_matrices):
        print(f'Action: {actions[i]}')
        print(matrix)
        print()

pipeline = Pipeline([
    ('preprocess', FunctionTransformer(load_data, kw_args={'actions': actions})),
    ('model', FunctionTransformer(build_model)),
    ('evaluate', FunctionTransformer(evaluate_model, kw_args={'actions': actions}))
])

# Fit the pipeline
pipeline.fit()
# y_pred = pipeline.predict(actions)

# model = pipeline.named_steps['model']
# X_test = pipeline.named_steps['preprocess'][-2]
# y_test = pipeline.named_steps['preprocess'][-1]

# evaluate_model(model, X_test, y_test, actions)
