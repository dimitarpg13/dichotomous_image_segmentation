import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown
import os

import requests
import matplotlib.pyplot as plt
from io import BytesIO

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown
import os

import requests
import matplotlib.pyplot as plt
from io import BytesIO

from IS_Net.data_loader_cache import normalize, im_reader, im_preprocess
from IS_Net.models.isnet import ISNetGTEncoder, ISNetDIS

from huggingface_hub import hf_hub_download
import shutil

HUGGING_FACE_DIS_MODEL_REPO = 'dimitarpg13/DIS'

DIS_PRETRAINED_MODEL_FILE_NAME = "isnet-general-use.pth"

LOCAL_SAVED_MODELS_FOLDER = "./saved_models"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def maybe_download_pretrained_model():
    """
    Downloads the pretrained model and stored it locally if it doesn't exist.'
    """
    if not os.path.exists(LOCAL_SAVED_MODELS_FOLDER):
        os.mkdir(LOCAL_SAVED_MODELS_FOLDER)
        local_temp_file = hf_hub_download(repo_id=HUGGING_FACE_DIS_MODEL_REPO, filename=DIS_PRETRAINED_MODEL_FILE_NAME)
        shutil.copy(local_temp_file, LOCAL_SAVED_MODELS_FOLDER)


class GOSNormalize(object):
    """
    Normalize the Image using torch.transforms
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


class SegmentationManager:
    def __init__(self):
        pass

