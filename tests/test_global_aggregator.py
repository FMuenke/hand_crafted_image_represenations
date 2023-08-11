import unittest
import numpy as np
from classic_image_classification.machine_learning.global_aggregator import GlobalAggregator


def create_test_descriptors():
    test_desc = [
        0 * np.ones((3, 8)),
        1 * np.ones((5, 8)),
        4 * np.ones((9, 8)),
        8 * np.ones((1, 8)),
        None,
    ]
    expected_res = [
        0 * np.ones((1, 8)),
        1 * np.ones((1, 8)),
        4 * np.ones((1, 8)),
        8 * np.ones((1, 8)),
        0 * np.ones((1, 8)),
    ]
    return test_desc, expected_res


class TestGlobalAggregator(unittest.TestCase):

    def test_aggregate_global_avg(self):
        aggregator = GlobalAggregator("global_avg")
        descriptors, expected_result = create_test_descriptors()
        aggregator.fit(descriptors)
        aggregated = aggregator.transform(descriptors)
        np.testing.assert_array_equal(aggregated, expected_result)

    def test_aggregate_global_max(self):
        aggregator = GlobalAggregator("global_max")
        descriptors, expected_result = create_test_descriptors()
        aggregator.fit(descriptors)
        aggregated = aggregator.transform(descriptors)
        np.testing.assert_array_equal(aggregated, expected_result)

    def test_fit(self):
        descriptors, _ = create_test_descriptors()
        aggregator = GlobalAggregator("global_avg")
        aggregator.fit(descriptors)
        self.assertEqual(aggregator.n_features, 8)

        aggregator = GlobalAggregator("global_max")
        aggregator.fit(descriptors)
        self.assertEqual(aggregator.n_features, 8)

        aggregator = GlobalAggregator("global_std")
        aggregator.fit(descriptors)
        self.assertEqual(aggregator.n_features, 16)

    def test_transform(self):
        descriptors, _ = create_test_descriptors()
        aggregator = GlobalAggregator("global_avg")
        aggregator.fit(descriptors)
        transformed = aggregator.transform(descriptors)
        self.assertEqual(len(transformed), 5)
        self.assertEqual(transformed[0].shape, (1, 8))

        descriptors, _ = create_test_descriptors()
        aggregator = GlobalAggregator("global_std")
        aggregator.fit(descriptors)
        transformed = aggregator.transform(descriptors)
        self.assertEqual(len(transformed), 5)
        self.assertEqual(transformed[0].shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
