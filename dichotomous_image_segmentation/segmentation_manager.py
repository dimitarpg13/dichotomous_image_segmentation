import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import os

import requests
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


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_image(im_path, hypar):
    if im_path.startswith("http"):
        im_path = BytesIO(requests.get(im_path).content)

    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)  # make a batch of image, shape


def build_model(hypar, device):
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if hypar["restore_model"] != "":
        net.load_state_dict(torch.load(hypar["model_path"] + "/" + hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()
    return net


def get_center_of_mass(mask):
    """
    Computes the center of mass of a mask
    :param mask:
    :return: tuple with x and y coordinates of the center of mass of the mask
    """
    x_coord = 0.0
    y_coord = 0.0
    tot_mass = 0.0
    for col_idx, col in enumerate(mask):
        for row_idx, pixel in enumerate(col):
            x_coord += row_idx * pixel
            y_coord += col_idx * pixel
            tot_mass += pixel

    return tuple([round(x_coord / tot_mass), round(y_coord / tot_mass)])


class SegmentationManager:
    def __init__(self, device):
        self.device = device

        # parameters for inference
        self.hypar = {"model_path": LOCAL_SAVED_MODELS_FOLDER,
                      "restore_model": DIS_PRETRAINED_MODEL_FILE_NAME,
                      "interm_sup": False,
                      "model_digit": "full",
                      "seed": 0,
                      "cache_size": [1024, 1024],
                      "input_size": [1024, 1024],
                      "crop_size": [1024, 1024],
                      "model": ISNetDIS()}  # parameters for inferencing

        self.net = build_model(self.hypar, self.device)

    def predict_mask_from_tensor(self, inputs_val, shapes_val):
        """
        Given an Image, predict the mask
        :inputs_val: Image
        :shapes_val: image size
        :return: mask
        """
        self.net.eval()

        if self.hypar["model_digit"] == "full":
            inputs_val = inputs_val.type(torch.FloatTensor)
        else:
            inputs_val = inputs_val.type(torch.HalfTensor)

        inputs_val_v = Variable(inputs_val, requires_grad=False).to(self.device)  # wrap inputs in Variable

        ds_val = self.net(inputs_val_v)[0]  # list of 6 results

        # we want the first one which is the most accurate prediction
        pred_val = ds_val[0][0, :, :, :]  # B x 1 x H x W

        # recover the prediction spatial size to the original image size
        pred_val = torch.squeeze(
            F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

        ma = torch.max(pred_val)
        mi = torch.min(pred_val)
        pred_val = (pred_val - mi) / (ma - mi)  # max = 1

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)  # it is the mask we need

    def predict_from_file(self, image_path, image_name, image_extension):
        """
        :param image_path: str
        :param image_name: str
        :param image_extension: str ('JPEG' or 'PNG')
        """
        image_name = "smiling-man"
        full_image_path = image_path + image_name + "." + image_extension

        image_tensor, orig_size = load_image(full_image_path, self.hypar)
        mask = self.predict_mask_from_tensor(image_tensor, orig_size)

        return mask
