import unittest
import os
from pathlib import Path
from dichotomous_image_segmentation.segmentation_manager import SegmentationManager, get_center_of_mass, DEVICE

DIS_HOME = '/'.join(Path(os.path.abspath(os.path.dirname(__file__))).parts[:-2]).replace('/','',1)
DATASET_PATH = 'demo_datasets/your_dataset/'


class TestSegmentationManager(unittest.TestCase):
    def setUp(self):
        self.seg_manager = SegmentationManager(DEVICE)

    def test_predict_from_file(self):
        image_path = DIS_HOME + "/" + DATASET_PATH
        image_name = 'smiling-man'
        mask = self.seg_manager.predict_from_file(image_path, image_name, 'jpeg')
        (x_coord_center_of_mass, y_coord_center_of_mass) = get_center_of_mass(mask)
        self.assertEqual(x_coord_center_of_mass, 300)
        self.assertEqual(y_coord_center_of_mass, 360)


if __name__ == '__main__':
    unittest.main()
