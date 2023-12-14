import os
import cv2
import random
import threading
import numpy as np
import mediapipe as mp
import pipelines.pipes.Data_prepare_pipeline as prepareD
import pipelines.pipes.training_pipeline as trainM
from .models import *
from .forms import *
from joblib import dump, load
import sqlite3
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators import gzip
from data.upload_retraining_json_to_db import DataHandler
import json
from django.http import JsonResponse
from datetime import date
from tensorflow.keras.models import Sequential
from django.core.files.base import ContentFile
from django.core.files.base import File
from tensorflow.keras.layers import LSTM, Dense, Dropout
from plyer import notification
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



# Home
def home(request):
    return render(request, "home.html", {})


# Explanation
def project_explanation(request):
    return render(request, "explanation.html", {})


# Register user
def register_user(request):
    user_form = SignUpForm()
    # Check if the form was filled in
    if request.method == "POST":
        # Take input from the webpage and add to SignUpForm
        user_form = SignUpForm(request.POST)
        if user_form.is_valid():
            user_form.save()
            username = user_form.cleaned_data['username']
            password = user_form.cleaned_data['password1']
            # Log in user
            user = authenticate(username=username, password=password)
            login(request, user)
            messages.success(request, ("You have been registered successfully."))
            return redirect("home")
        else:
            messages.success(request, ("There was an error. Please try to again."))
            return redirect("register")
    else:
        return render(request, "register.html", {'user_form':user_form})


# Login
def login_user(request):
    # If the form was filled in and button clicked
    if request.method == "POST":
        # Get input from login form
        username = request.POST['username']
        password = request.POST['password']
        # Use Django authenticate system to pass in username and password from the form
        user = authenticate(request, username=username, password=password)
        # Login if the form was filled in
        if user is not None:
            login(request, user)
            messages.success(request, ("You have been logged in."))
            if user.is_superuser:
                return redirect('training')
            else:
                return redirect('translations')
        # Redirect if login was not successful 
        else:
            messages.success(request, ("There was an error. Please try to again."))
            return redirect('login')
    # If the form was not filled in, show the login page    
    else:
        return render(request, "login.html", {'navbar': 'login'})


# Logout
def logout_user(request):
    logout(request)
    messages.success(request, ("You have been logged out."))
    return redirect('home')


# Retraining functionality
def training_functionality():
    '''
    notification.notify(
    title='Retraining',
    message='The model is being retrained. Be patient!',
    app_name='ASL',
    )
    '''
    data = prepareD.data_pipeline.fit_transform(None)
    ##run some tests
    data2 = data
    result = trainM.train_pipeline.fit_transform(None)

    trained_model = result['model']
    accuracy = result['accuracy']
    train_accuracy = result['train_accuracy']
    trained_model.save('asl_translator_gui/trained_models/{}.h5'.format(random.randint(1000,9999)))
    # abs_path = os.path.abspath('models')
    print(f"accuracy {accuracy}")
    print(f" train accuracy {train_accuracy}")
    # abs_path = os.path.abspath("data/retrained-model")
    # dump(trained_model,f'{abs_path}.joblib'.format(random.randint(1000, 9999)))
    # dump(trained_model, f'{abs_path}/{random.randint(1000, 9999)}.joblib')
    path_to_the_h5 = trained_model.name
    return path_to_the_h5, accuracy, train_accuracy
    
def training(request):
    current_user = request.user
    training_list = Training.objects.all()
    tr_upload_form = UploadTrainingForm()

    if request.method == 'POST':
        tr_upload_form = UploadTrainingForm(request.POST, request.FILES)

        if tr_upload_form.is_valid():

            #uploaded_file = request.FILES['tr_input_file']
            instance = tr_upload_form.save(commit=False)
            instance.tr_input_user = current_user
            instance.save()
            trained_model, accuracy, train_accuracy = training_functionality()
            input = Training_input.objects.get(tr_input_id = instance.tr_input_id)
            tr_input_file = input
            training_date = date.today()
            training_accuracy = int(train_accuracy * 100)
            testing_accuracy = int(accuracy * 100)
            model_weights = trained_model
            is_deployed = False
            new_record = Training(tr_input_file=tr_input_file, training_date=training_date, training_accuracy=training_accuracy, testing_accuracy=testing_accuracy, model_weights=model_weights, is_deployed=is_deployed)
            new_record.save()
        else:
            tr_error_messages = tr_upload_form.errors.values()
            return render(request, "training.html", {'training_list': training_list, 'tr_upload_form': tr_upload_form, 'tr_error_messages': tr_error_messages})
    return render(request, "training.html", {'training_list': training_list, 'tr_upload_form': tr_upload_form})

# History of user's translations
def translations(request):
    # Get translations for the current user
    current_user = request.user
    translation_list = Translation_input.objects.all()
    output_list = Translation_output.objects.all()  
    # Upload file
    upload_form = UploadForm()
    if request.method == "POST":
        upload_form = UploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            instance = upload_form.save(commit=False)
            instance.input_user = current_user    # Assign upload file with currently logged in user
            instance.save()
            translateFile(instance.input_id)
        else:
            error_messages = upload_form.errors.values()
            return render(request, "translations.html", {'translation_list': translation_list, 'output_list':output_list, 'upload_form':upload_form, 'error_messages':error_messages})
    return render(request, "translations.html", {'translation_list': translation_list, 'output_list':output_list, 'upload_form':upload_form})

# Translate file
def translateFile(input_id):
    input = Translation_input.objects.get(input_id = input_id)
    input_file = input.input_file
    print(str(input_file))
    output_file_path = f'media/output/{input.file_name()[:-4]}.txt'
    print(f'This is the output file in translation{output_file_path}')
    output = Translation_output(output_user=input.input_user, output_source=input, output_file=output_file_path)
    output.save()
    generate_output(input_file, output_file_path)

def deploy(request, model_id):
  training_list = Training.objects.all()
  training = Training.objects.get(model_id=model_id)
  active_model = Training.objects.get(is_deployed = "True")
  active_model.is_deployed = "False"
  active_model.save()
  training.is_deployed = "True"
  training.save()
  return redirect('training')

@gzip.gzip_page
def live(request):
    try:
        return StreamingHttpResponse(gen(), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'live.html')

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
    lh_has_values = any(lh)
    rh_has_values = any(rh)
    return np.concatenate([pose, face, lh, rh]), lh_has_values, rh_has_values

def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic 
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_holistic = mp.solutions.holistic 
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def load_model():
    # Load the model
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
    training = Training.objects.get(is_deployed = "True")
    deployed_model = training.model_weights
    absolute_path = os.path.abspath("media/" + str(deployed_model))
    # model architecture
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(27,1662)))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(np.array(actions).shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print(absolute_path)
    model.load_weights(absolute_path)
    return model, actions
    
def evaluate(model):
    DATA_PATH_TEST = os.path.join(os.path.abspath('./data/test_set')) 
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
    label_map = {label: num for num, label in enumerate(actions)}
    sequences_test, labels_test = [], []
    for action_test in actions:
        videos = os.listdir(os.path.abspath("./data/test_set") + "/" + "{}".format(action_test))
        if ".DS_Store" in videos:
            videos.remove(".DS_Store")
        for sequence in videos:
            window = []
            if sequence != ".DS_Store":
                #count the number of np arrays (frames) this video has
                number_of_f = os.listdir(os.path.abspath("./data/test_set") + "/" + "{}".format(action_test) + "/" + sequence)
                f_size = len(number_of_f)

                if f_size == 30:
                    for frame_num in range(3, 30):
                        res = np.load(
                            os.path.join(DATA_PATH_TEST, action_test, sequence, "{}.npy".format(frame_num)))
                        window.append(res)
                    
                sequences_test.append(window)
                labels_test.append(label_map[action_test])
                    
    K = np.array(sequences_test)
    P = to_categorical(labels_test).astype(int)
    _, X_test_set, _, y_test_set = train_test_split(K, P, test_size=0.99)

    _, goz = model.evaluate(X_test_set, y_test_set)
    print('goz: ', goz)

def gen():
    print("inside the gen function")
    model, actions = load_model()
    evaluate(model)
    # 1. New detection variables
    sequence = []           # placeholder for 29 frames which makes up a video/sequence
    sentence = []           # the tranlation result
    predictions = []        # what the model predicts
    threshold = 0.70       # how confident should the resulted prediction be so that we present/use it

    mp_holistic = mp.solutions.holistic # Holistic model
    
    # give the source of video - 0 for the camera and video path for uploaded videos
    cap = cv2.VideoCapture(0)
    # the tool we need for extracting keypoints and drawing
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
        #while True:
        while cap.isOpened():
            # Process the frame (resize, preprocess, etc.)
            frames_since_hands = 0
            
            ret, frame = cap.read()     #reads frames every interation 
            #print("hereeeee")
            #if not ret:
             #   print('failed to read frame from the video!')
              #  break
            # Make detections
            image, results = mediapipe_detection(frame, holistic)   
            print(results)
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            if any([results.right_hand_landmarks, results.left_hand_landmarks]):
             #Hands are detected, increment the frame counter
                frames_since_hands += 1
            else:
            # No hands detected, reset the frame counter
                frames_since_hands = 0
        
            # 2. Prediction logic
            if frames_since_hands >=3:
                keypoints, lh, rh = extract_keypoints(results)
                #if lh or rh:
                sequence.append(keypoints)
                sequence = sequence[-27:]
        
            # if we have seen 29 frames then
            if len(sequence) == 27:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                print(res[np.argmax(res)])
                predictions.append(np.argmax(res))

            #3. Viz logic
                if np.unique(predictions[-12:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                            
                    # Limit to last 5 words
                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                #sequence = []
            
            # draw the output on the screen
            text_color = (0, 255, 255)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_4)
            #cv2.imshow('OpenCV Feed', image)
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


            
def generate_output(camera, output_file):
    # Load the model
    print("inside the generate_output function.")
    model, actions = load_model()
    # 1. New detection variables
    sequence = []           # placeholder for 29 frames which makes up a video/sequence
    sentence = []           # the translation result
    predictions = []        # what the model predicts
    threshold = 0.7        # how confident should the resulted prediction be so that we present/use it

    mp_holistic = mp.solutions.holistic # Holistic model
    # give the source of video - 0 for the camera and video path for uploaded videos
    absolute_path_to_camera = os.path.abspath("media/" +str(camera))
    cap = cv2.VideoCapture(absolute_path_to_camera)
    # the tool we need for extracting keypoints and drawing
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if(cap.isOpened()==False):
            print("the camera is not open")
        while cap.isOpened():
            #frames_since_hands = 0
            ret, frame = cap.read()     #reads frames every iteration 
            if not ret:
                print('failed to read frame from the video!')
                break
            # Make detections
            image, results = mediapipe_detection(frame, holistic)   
            print(results)
            # Draw landmarks
            draw_styled_landmarks(image, results)
        
            
            if any([results.right_hand_landmarks, results.left_hand_landmarks]):
             #Hands are detected, increment the frame counter
                frames_since_hands += 1
            else:
            # No hands detected, reset the frame counter
                frames_since_hands = 0
            
            # 2. Prediction logic
            if frames_since_hands >=3:
                keypoints, lh, rh = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-27:]

            # if we have seen 29 frames then
            if len(sequence) == 27:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                print(res[np.argmax(res)])
                predictions.append(np.argmax(res))

        
            #3. Viz logic
                if np.unique(predictions[-15:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

            # draw the output on the screen
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    # Save detected words to a text file
    with open(output_file, 'w') as file:
        # file.write(' '.join(sentence))
        file.write(str(sentence))
        file.close()
        
    cap.release()
    cv2.destroyAllWindows()
