import unittest
import numpy as np
from hand_crafted_image_representations.machine_learning.classifier import Classifier


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_train = np.random.rand(200, 10)
        cls.y_train = np.random.randint(0, 2, size=200)
        cls.x_test = np.random.rand(20, 10)
        cls.y_test = np.random.randint(0, 2, size=20)

    def test_fit(self):
        classifier = Classifier()
        classifier.opt = {"type": "lr"}
        classifier.fit(self.x_train, self.y_train)
        self.assertIsNotNone(classifier.classifier)

    def test_predict(self):
        classifier = Classifier()
        classifier.opt = {"type": "lr"}
        classifier.fit(self.x_train, self.y_train)
        predictions = classifier.predict(self.x_test)
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_evaluate(self):
        classifier = Classifier()
        classifier.opt = {"type": "lr"}
        classifier.fit(self.x_train, self.y_train)
        f1_macro = classifier.evaluate(self.x_test, self.y_test, print_results=False)
        self.assertIsInstance(f1_macro, float)

    def test_new(self):
        classifier = Classifier()
        for clf in ["lr", "knn_3", "mlp_x", "rf"]:
            classifier.opt = {"type": clf}
            classifier.new()
            self.assertIsNotNone(classifier.classifier)


if __name__ == "__main__":
    unittest.main()
