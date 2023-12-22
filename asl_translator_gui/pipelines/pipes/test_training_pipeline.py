import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from pipelines.pipes.training_pipeline import TrainModelTransformer
from tensorflow.keras.utils import to_categorical
from sklearn.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Example data
        actions = ["hello"]
        target_file_count = 29

        label_map = {label: num for num, label in enumerate(actions)}
        sequences, labels = [], []

        for action in actions:
            for _ in range(50):  # Create 50 example sequences
                window = np.random.rand(target_file_count, 1662)
                sequences.append(window)
                labels.append(label_map[action])

        Z = np.array(sequences)
        y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.2)

        self.dummy_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,  
            'y_test': y_test,   
            
        }
        
    def test_train_model_transformer_fit_transform(self):
        pipeline = TrainModelTransformer(actions=["hello"], target_file_count=29, epochs = 20)
        result = pipeline.fit_transform(self.dummy_data)
        model = result['model']
        accuracy = result['accuracy']
        train_accuracy = result['train_accuracy']

        # assertions
        self.assertIsNotNone(model)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(train_accuracy, float)

    def test_pipeline_fit_transform(self):
        pipeline = Pipeline([
            ('preprocessAndFit', TrainModelTransformer(actions=["hello"], target_file_count=29, epochs= 20))
        ])

        result = pipeline.fit_transform(self.dummy_data)
        model = result['model']
        accuracy = result['accuracy']
        train_accuracy = result['train_accuracy']

        # Assertions
        self.assertIsNotNone(model)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(train_accuracy, float)

if __name__ == '__main__':
    unittest.main()
