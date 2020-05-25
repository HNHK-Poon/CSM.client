import sys
sys.path.remove('/home/poon/PycharmProjects/CSM.client/src/testing')
sys.path.extend(['/home/poon/PycharmProjects/CSM.client/src'])
print(sys.path)
import cv2
import numpy as np
import json
import requests
from testing.image_processing import ImageProcessor


class demo:
    def __init__(self):
        self.processor = ImageProcessor()

    def process_image(self, url):
        np_image = self.processor.process_image(url)
        return np_image

    def api_request(self, input_image):
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


if __name__ == '__main__':
    app = demo()
    url = "/home/poon/Downloads/testing2/9-1.jpeg"
    np_image = app.process_image(url)
    np_image = np_image.reshape(1, 128, 128, 3)
    print(np_image.shape)
    predicted = app.api_request(np_image)
    print('predicted: {}'.format(np.argmax(predicted)))


