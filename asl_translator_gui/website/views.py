from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse
from .models import *
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from joblib import load
import numpy as np
import mediapipe as mp

# Home
def home(request):
    return render(request, "home.html", {})

# About
def about(request):
    return render(request, "about.html", {})

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
            return redirect('training')
        # Redirect if login was not successful 
        else:
            return redirect('login')
    # If the form was not filled in, show the login page    
    else:
        return render(request, "login.html", {})

# Logout
def logout_user(request):
    logout(request)
    return redirect('home')

# Model training
def training(request):
    training_list = Training.objects.all()
    return render(request, "training.html", {'training_list': training_list})







# Holistics for the drawing of keypoints
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# The labels/words we can predict
actions = np.array(['hello', 'nice', 'meet', 'you'])

# Load the model
model = load('media/media/models/V_0.joblib')



@gzip.gzip_page
def live(request):
    try:
        cam = VideoCamera()
        # calls the live video feed method and pass it to be rendered
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
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
    return np.concatenate([pose, face, lh, rh])

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
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



#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        #_, jpeg = cv2.imencode('.jpg', image)
        #return jpeg.tobytes()
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    # 1. New detection variables
    sequence = []           # placeholder for 29 frames which makes up a video/sequence
    sentence = []           # the tranlation result
    predictions = []        # what the model predicts
    threshold = 0.2         # how confident should the resulted prediction be so that we present/use it

    # give the source of video- 0 for the camera and video path for uploaded videos
    #cap = cv2.VideoCapture(0)
    # the tool we need for extracting keypoints and drawing
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = camera.get_frame()     #reads frames every interation 
            print("hereeeee", type(frame), frame.shape)

            # Make detections
            image, results = mediapipe_detection(frame, holistic)   

            # Draw landmarks
            draw_styled_landmarks(image, results)
        
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.insert(0,keypoints)
            sequence = sequence[:29]
        
            # if we have seen 29 frames then
            if len(sequence) == 29:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

            #3. Viz logic
            if predictions and np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            # only the last 5 words
            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            #image = prob_viz(res, actions, image, colors)
       
            # draw the output on the screen
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # save the result and pass it to the streem
            _, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
