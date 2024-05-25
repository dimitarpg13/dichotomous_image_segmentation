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


class SegmentationManager:
    def __init__(self):
        pass

