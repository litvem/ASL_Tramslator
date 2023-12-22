import unittest
import os
import numpy as np
import cv2
import cv2.ml
import mediapipe as mp
import pipelines.pipes.Data_prepare_pipeline as DP
import shutil


class VideoProcessorTests(unittest.TestCase):
    def setUp(self):
        self.mp_holistic = mp.solutions.holistic

    def tearDown(self):
        pass


    def test_mediapipe_detection(self):
        # Mock data
        mp_holistic = mp.solutions.holistic
        image = cv2.cvtColor(cv2.imread('pipelines/pipes/test2/0.png'), cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        actual_result = model.process(image)
        _,results = DP.mediapipe_detection(cv2.imread("pipelines/pipes/test2/0.png"), model)
        results_pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        results_face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        results_lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        results_rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        actual_pose =  np.array([[res.x, res.y, res.z, res.visibility] for res in actual_result.pose_landmarks.landmark]).flatten() if actual_result.pose_landmarks else np.zeros(33*4)
        actual_face = np.array([[res.x, res.y, res.z] for res in actual_result.face_landmarks.landmark]).flatten() if actual_result.face_landmarks else np.zeros(468*3)
        actual_lh = np.array([[res.x, res.y, res.z] for res in actual_result.left_hand_landmarks.landmark]).flatten() if actual_result.left_hand_landmarks else np.zeros(21*3)
        actual_rh = np.array([[res.x, res.y, res.z] for res in actual_result.right_hand_landmarks.landmark]).flatten() if actual_result.right_hand_landmarks else np.zeros(21*3)
        # Assertions
        self.assertIsNotNone(actual_pose)
        self.assertIsNotNone(actual_face)
        self.assertIsNotNone(actual_lh)
        self.assertIsNotNone(actual_rh)
        self.assertEqual(len(actual_pose), len(results_pose))
        self.assertEqual(len(actual_face), len(results_face))
        self.assertEqual(len(actual_lh), len(results_lh))
        self.assertEqual(len(actual_rh), len(results_rh))

    def test_extract_keypoints(self):
        mp_holistic = mp.solutions.holistic
        image = cv2.cvtColor(cv2.imread('pipelines/pipes/test2/0.png'), cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB 
        model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        image, results = DP.mediapipe_detection(image, model)
        keypoints  = DP.extract_keypoints(results)

        # # Assertions
        self.assertEqual(len(keypoints), 1662)
        
    def test_save_frames(self):
        # Create test data
        file_name = 'pipelines/pipes/test2/videos/0.mp4'
        sequence_length = 20
        path_to_np_video = 'pipelines/pipes/test2/MP_data'
        DP.save_frames(file_name, sequence_length, path_to_np_video)

        # Assertions
        self.assertEqual(len(os.listdir(path_to_np_video)), sequence_length)

    def test_save_videos(self):
        # Create test data 
        data_point = {
            "clean_text": "hello",
            "start_time": 0,
            "end_time": 1.735,
            "url": "https://www.youtube.com/watch?v=p8OYydc3WQM"
        }
        videos_address = 'pipelines/pipes/test2/videos'
        np_address = os.path.join('pipelines/pipes/test2/MP_data') 
        video_id = 1
        sequence_length = 20
        result = DP.save_videos(data_point, videos_address, sequence_length, np_address, video_id)

        # Assertions
        self.assertTrue(result)
        self.assertTrue(os.path.exists(np_address))
        self.assertTrue(os.path.exists(os.path.join(np_address, 'hello')))
        self.assertTrue(os.path.exists(os.path.join(np_address, 'hello', '1')))
        self.assertTrue(os.path.exists(os.path.join(np_address, 'hello', '1', '0.npy')))