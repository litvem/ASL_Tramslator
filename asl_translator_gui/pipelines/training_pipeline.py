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

DATA_PATH = os.path.join('test/MP_Data') 
actions = ["hello", "nice", "meet", "you"]


def load_data(X,actions, target_file_count=29):
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        videos = os.listdir("test/MP_Data" + "/" + "{}".format(action))
        if ".DS_Store" in videos:
            videos.remove(".DS_Store")

        random_videos = random.sample(videos, 33)

        for sequence in random_videos:
            window = []
            if sequence != ".DS_Store":
                number_of_f = os.listdir("test/MP_Data" + "/" + "{}".format(action) + "/" + sequence)
                f_size = len(number_of_f)

                if f_size <= target_file_count:
                    for frame_num in range(target_file_count):
                        res = np.load(
                            os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                        window.append(res)
                elif f_size >= 60:
                    for frame_num in range(0, f_size - 2, 2):
                        res = np.load(
                            os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))
                        window.append(res)
                else:
                    random_numbers = random.sample(range(0, 30), target_file_count)
                    random_numbers.sort()
                    for frame_num in random_numbers:
                        res = np.load(
                            os.path.join(DATA_PATH, action, sequence, "{}.npy".format(frame_num)))    
                        window.append(res)

                sequences.append(window)
                labels.append(label_map[action])
                
    Z = np.array(sequences)
    print(Z.shape)
    y = to_categorical(labels).astype(int)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.2)

    print(X_train.shape)
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(29,1662)))
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
    model.add(Dense(np.array(actions).shape[0], activation='softmax'))
    tb_callback = TensorBoard(log_dir=log_dir)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=10, callbacks=[tb_callback])
    res = model.predict(X_test)
    for i in range(28):
        print("actual: ", actions[np.argmax(res[i])], " - result: ", actions[np.argmax(y_test[i])])


pipeline = Pipeline([
    ('preprocessAndFit', FunctionTransformer(func=load_data, kw_args={'actions': actions}))
])


model = pipeline.fit_transform(None)

