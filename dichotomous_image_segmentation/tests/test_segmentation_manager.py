import unittest
from dichotomous_image_segmentation.segmentation_manager import SegmentationManager, DEVICE


class SegmentationManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.seg_manager = SegmentationManager(DEVICE)

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
