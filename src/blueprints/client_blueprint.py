from config.ConfigValue import ConfigValue
from flask import Blueprint, request, render_template, Markup, redirect
import json
import scipy.misc
import numpy as np
import requests
import os
from imgaug import augmenters as iaa
import imgaug as ia
import base64
import re
from PIL import Image
from io import BytesIO
import cv2
from testing.image_processing import ImageProcessor
from config.ConfigValue import ConfigValue

np.random.bit_generator = np.random._bit_generator

import matplotlib as mpl
import matplotlib.pyplot as plt

filenames = os.listdir('/home/poon/Downloads/testing/')
image_urls = ['/home/poon/Downloads/testing/'+filename for filename in filenames if filename.endswith(".jpeg")]
predict_image_url = '/home/poon/Downloads/test.jpeg'
client_blueprint = Blueprint("client_blueprint", __name__)
config = ConfigValue()

@client_blueprint.route("/app/index.html", methods=["GET", "POST"])
def show_default_page():
    if request.method == "POST":
        pass
    default_page_template = render_template("index.html")
    return Markup(default_page_template)


@client_blueprint.route("/", methods=["GET", "POST"])
def redirect_default_page():
    if request.method == "POST":
        pass
    return redirect("/app/index.html")


@client_blueprint.route("/app", methods=["GET", "POST"])
def redirect_default_app_page():
    if request.method == "POST":
        pass
    return redirect("/app/index.html")


@client_blueprint.route('/api/predict', methods=['GET', 'POST'])
def get_prediction_result():
    config = ConfigValue()
    imageProcessor = ImageProcessor()
    imageProcessor.show_processes_count()
    predict_mode = 'grayscale'
    data = request.form
    #print(data)
    print("decoding...")
    image_data = re.sub('^data:image/.+;base64,', '', data['img'])
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    # print("decoded", np.array(image).shape)
    process_result = imageProcessor.process_image(np.array(image))
    # print(process_result[0], imageProcessor.show_processes_count())
    if process_result is not None:
        np_image = process_result[0]
        crop_params = process_result[1]
        np_image = np_image.reshape(1,128,128,3)
        response = api_request(np_image)[0]
        result = process_response(response, crop_params)
        return json.dumps(result)

    result = {
        "isItem": False,
    }
    return json.dumps(result)

def process_response(response, crop_params):
    softmax = np.exp(response) / sum(np.exp(response))
    sorted_class = np.argsort(response)[::-1]
    sorted_prob = softmax[sorted_class]
    sorted_prob_bool = sorted_prob > 1e-2
    sorted_prob = sorted_prob[sorted_prob_bool]
    sorted_class = sorted_class[:len(sorted_prob)]
    sorted_name = []
    for _class in sorted_class:
        sorted_name.append(config.get_value('CLASSNAME', str(_class)),)
    print("output: \n{} \nsoftmax:\n{}\n".format(sorted_class, sorted_prob))

    result = {
        "top": crop_params['top'],
        "bottom": crop_params['bottom'],
        "left": crop_params['left'],
        "right": crop_params['right'],
        "isItem": True,
        "name": sorted_name,
        "class": sorted_class.tolist(), #str(np.argmax(response)),
        "prob": sorted_prob.tolist()
    }
    return result

#for predict_image_url in image_urls:
    # np_image_color, np_image_grayscale = process_image(predict_image_url)
    # #os.remove(predict_image_url)
    # #print('image to be predict: {}'.format(np_image))
    #
    # csm_color_url = "http://localhost:8501/v1/models/csm_color:predict"
    # json_data_color = {
    #     "signature_name": "predict",
    #     "inputs": {
    #         "imageInput": np_image_color.tolist(),
    #         "isTraining": False
    #     }
    # }
    # response_color = requests.post(url=csm_color_url, json=json_data_color)
    #
    # csm_grayscale_url = "http://localhost:8501/v1/models/csm_grayscale:predict"
    # json_data_grayscale = {
    #     "signature_name": "predict",
    #     "inputs": {
    #         "imageInput": np_image_grayscale.tolist(),
    #         "isTraining": False
    #     }
    # }
    # response_grayscale = requests.post(url=csm_grayscale_url, json=json_data_grayscale)
    #
    # output_color = json.loads(response_color.text)["outputs"]
    # output_grayscale = json.loads(response_grayscale.text)["outputs"]
    # output_blend = np.add(output_color, output_grayscale)
    #
    # if predict_mode == 'color':
    #     predict_class = np.argmax(output_color)
    #     confidence_level = output_color[0][np.argmax(output_color)]*100
    # elif predict_mode == 'grayscale':
    #     predict_class = np.argmax(output_grayscale)
    #     confidence_level = output_grayscale[0][np.argmax(output_grayscale)] * 100
    # else:
    #     predict_class = np.argmax(output_blend)
    #     confidence_level = output_blend[0][np.argmax(output_blend)] * 100

    # result = {
    #     "predicted_class": predicted, #config.get_value('CLASSNAME', str(predict_class)),
    #     "confidence_level": '89' #str(int(confidence_level)-1)
    # }
    # #print(predict_image_url, result)
    # return json.dumps(result)

def api_request(input_image):
    csm_grayscale_url = "http://localhost:8501/v1/models/csm:predict"
    json_data = {
        "signature_name": "predict",
        "inputs": {
            "imageInput": input_image.tolist(),
            "isTraining": False
        }
    }
    response = requests.post(url=csm_grayscale_url, json=json_data)
    return json.loads(response.text)["outputs"]


# def process_image(predict_image_url):
#     np_image = scipy.misc.imread(predict_image_url, mode='RGB')
#     np_image = scipy.misc.imresize(np_image, (128, 128))
#     np_image = augmentation(np_image)
#     np_image = np_image.astype(np.float32)
#     np_image /= 127.5
#     np_image -= 1.
#     np_image_color = np_image.reshape(1, 128, 128, 3)
#
#     np_image_grayscale = scipy.misc.imread(predict_image_url, mode='L')
#     np_image_grayscale = scipy.misc.imresize(np_image_grayscale, (128, 128))
#     np_image_grayscale = augmentation(np_image_grayscale)
#     np_image_grayscale = np_image_grayscale.astype(np.float32)
#     np_image_grayscale /= 127.5
#     np_image_grayscale -= 1.
#     np_image_grayscale = np_image_grayscale.reshape(1, 128, 128, 1)
#     return np_image_color, np_image_grayscale
#
# def augmentation(image):
#     seq = iaa.Sequential([
#         iaa.Affine(rotate=(-2, 2)),
#         # iaa.AddToHueAndSaturation((-10, 10)), #-60, 60
#         iaa.AdditiveGaussianNoise(scale=(0, 0.2)),  # 5, 10
#         iaa.GammaContrast((0.9, 1.1)),  # 0.5, 1.5
#         # iaa.CoarseDropout((0.001, 0.01), size_percent=0.1)
#     ])
#     images_aug = seq.augment_image(image)
#     ia.imshow(images_aug)
#     return images_aug