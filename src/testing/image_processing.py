import cv2
from config.ConfigValue import ConfigValue
import numpy as np
import matplotlib.pyplot as plt
import math
import imgaug as ia
from imgaug import augmenters as iaa
np.random.bit_generator = np.random._bit_generator
from tqdm import tqdm

class ImageProcessor:
    def __init__(self):
        self.config = ConfigValue()
        self.crop_size = 128
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        self.max_k = 5
        self.min_cluster = 0.15
        self.padding_ratio = 1.3
        self.min_area = 0.03
        self.max_area = 0.45
        self.area_crop_size_ratio_step = 0.05
        self.padding_ratio_step = 0.02
        self.gamma = [1.3, 0.8, 1.5, 0.5]
        self.max_center_diff = 30

        self.count = {
            "success": 0,
            "contour failed": 0,
            "no contour": 0,
            "invalid center": 0,
            "invalid crop size": 0,
            "no params": 0,
            "crop failed": 0,
        }
        self.failed_img = []

    def read_img(self, img):
        # img = cv2.imread(path)
        # CSM client different with CSM
        img_to_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        gray = cv2.cvtColor(img_to_gray, cv2.COLOR_RGB2GRAY)
        # plt.imshow(img)
        # plt.show()
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), gray

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def get_contour_measurement(self, img):
        clone = img.copy()
        img_blur = cv2.GaussianBlur(img, (15, 15), 0)
        ret, thresh = cv2.threshold(img_blur, 127, 255, 0)
        # plt.imshow(thresh)
        # plt.show()
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        for contour in contours:
            try:
                # # approximte for circles
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                area = cv2.contourArea(contour) / (img.shape[0]*img.shape[1])
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = [cX, cY]
                # print("approx", approx)
                # print("area", area, cX, cY)
                # cv2.drawContours(clone, [contour], -1, (255, 0, 0), 3)
                # plt.imshow(clone)
                # plt.show()
                if ((len(approx) > 4) & (area < 0.15)):
                    # cv2.drawContours(clone, [contour], -1, (255, 0, 0), 3)
                    # plt.imshow(clone)
                    # plt.show()
                    contour_measurement = contour.reshape(-1, 2)
                    item_width = max(contour_measurement[:, 0]) - min(contour_measurement[:, 0])
                    item_height = max(contour_measurement[:, 1]) - min(contour_measurement[:, 1])
                    contour_list.append(contour)
                    return [center, item_height, item_width]
            except Exception as err:
                # cv2.drawContours(clone, [contour], -1, (255, 0, 0), 3)
                # plt.imshow(clone)
                # plt.show()
                self.count['contour failed'] += 1
                continue
        else:
            self.count['no contour'] += 1
            return None
        # contour = contour_list[0]
        # clone = img.copy()
        # for contour in contour_list:
        #     cv2.drawContours(clone, [contour], -1, (255, 0, 0), 3)
        #     plt.imshow(clone)
        #     plt.show()

    def process_img(self, img):
        # plt.imshow(img)
        # plt.show()
        k = 2
        while (k <= self.max_k):
            for gamma in self.gamma:
                image = self.adjust_gamma(img, gamma)
                shape = image.shape
                image = image.reshape((-1, 3))
                image = np.float32(image)
                _, label, (centers) = cv2.kmeans(image, k, None, self.criteria, 10, cv2.KMEANS_PP_CENTERS)
                centers = np.uint8(centers)
                label = label.flatten()
                segmented_image = centers[label.flatten()]
                segmented_image = segmented_image.reshape(shape)
                clusters = []
                for _k in range(k):
                    cluster = len(np.where(label == _k)[0])
                    cluster_ratio = cluster/len(label)
                    clusters.append(cluster_ratio)
                # print(clusters, label)
                if min(clusters) < self.min_cluster:
                    break
            else:
                k += 1
                continue
            break
        # plt.imshow(segmented_image)
        # plt.show()
        return segmented_image, label, clusters.index(min(clusters))

    def get_kmean_measurement(self, img, labels, k):
        heights = np.where(labels == k)[0] / img.shape[1]
        widths = np.where(labels == k)[0] % img.shape[1]
        left, right, top, bottom = int(np.amin(widths)), int(np.amax(widths)), int(np.amin(heights)), int(np.amax(heights))
        item_height, item_width = bottom-top, right-left
        center = [int((right + left) / 2), int((bottom + top) / 2)]
        return [center, item_height, item_width]

    def is_center_valid(self, kmean, contour):
        if kmean:
            kmean_center = kmean[0]
            if contour:
                contour_center = contour[0]
                diff = np.sum(np.absolute(np.subtract(kmean_center,contour_center)))
                # print('diff', diff, kmean_center, contour_center)
                if diff > self.max_center_diff:
                    self.count['invalid center'] += 1
                    return False
        return True

    def calculate_crop_size(self, img, measurements):
        if measurements:
            center, item_height, item_width = measurements
            padding_ratio = self.padding_ratio
            min_crop_area = (img.shape[0] * img.shape[1]) * self.min_area
            max_crop_area = (img.shape[0] * img.shape[1]) * self.max_area
            area_crop_size = int(math.sqrt(min_crop_area))
            max_area_crop_size = int(math.sqrt(max_crop_area))
            actual_crop_size = max(item_height, item_width)
            area_crop_size_ratio = 1
            if actual_crop_size < max_area_crop_size:
                while True:
                    if actual_crop_size < area_crop_size * 0.8:
                        crop_size = area_crop_size * area_crop_size_ratio
                    elif item_height > item_width:
                        crop_size = int(item_height/2*padding_ratio)
                    else:
                        crop_size = int(item_width/2*padding_ratio)

                    crop_params = {
                        "top": center[1]-crop_size,
                        "bottom": center[1]+crop_size,
                        "left": center[0]-crop_size,
                        "right": center[0]+crop_size
                    }

                    if (crop_params['left'] > 0 and
                        crop_params['top'] > 0 and
                        (img.shape[0]-crop_params['bottom']) > 0 and
                        (img.shape[1]-crop_params['right']) > 0):
                        break
                    else:
                        if actual_crop_size < area_crop_size * 0.8:
                            area_crop_size_ratio -= self.area_crop_size_ratio_step
                        else:
                            padding_ratio -= self.padding_ratio_step

                return crop_params
        self.count['invalid crop size'] += 1
        # self.failed_img.append(img)
        # print(actual_crop_size, max_area_crop_size)
        return False

    def convert_crop_size(self, img, params):
        print("image shape:{}".format(img.shape))
        converted_params = {
            "top": float(params["top"]) / img.shape[0],
            "bottom": float(params["bottom"]) / img.shape[0],
            "left": float(params["left"]) / img.shape[1],
            "right": float(params["right"]) / img.shape[1]
        }
        return converted_params

    def crop_img(self, img, crop_params):
        if crop_params:
            try:
                image_crop = img[crop_params['top']:crop_params['bottom'], crop_params['left']:crop_params['right']]
                # print(image_crop.shape)
                image_crop = cv2.resize(image_crop, (self.crop_size, self.crop_size))
                # plt.imshow(image_crop)
                # plt.show()

                # print(self.valid, self.loop)
                # if not self.valid:
                #     plt.imshow(image_crop)
                #     plt.show()
                # if self.loop:
                #     plt.imshow(image_crop)
                #     plt.show()
            except Exception:
                # print(img.shape, crop_params)
                # plt.imshow(img)
                # plt.show()
                self.count['crop failed'] += 1
                return None

            self.count['success'] += 1
            return image_crop
        return None

    def show_processes_count(self):
        print(type(self.count))
        print("Process info: {}".format(self.count))

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def augmentation_before(self, image):
        seq = iaa.Sequential([
            iaa.GammaContrast((0.8, 1.2)),  # 0.5, 1.5
        ])
        image_aug = seq.augment_image(image)
        return image_aug

    def augmentation(self, image):
        aug_size = int(self.config.get_value("TFRECORDS", "augment_size"))
        images = [image for i in range(aug_size)]
        seq = iaa.Sequential([
            #iaa.Affine(rotate=(-5, 5)),
            # iaa.AddToHueAndSaturation((-10, 10)), #-60, 60
            iaa.AdditiveGaussianNoise(scale=(0, 0.5)),  # 5, 10
            iaa.GammaContrast((0.8, 1.2)),  # 0.5, 1.5
            # iaa.CoarseDropout((0.001, 0.01), size_percent=0.1)
        ])
        images_aug = seq.augment_images(images)
        # ia.imshow(np.hstack(images_aug))
        return images_aug

    def format_image(self, np_image):
        np_image = np_image.astype(np.float32)
        np_image /= 127.5
        np_image -= 1.
        # np_image = np.reshape(np_image, (128, 128, 3))
        return np_image

    def process_images(self, jpgs, label):
        augment = True
        image_array = []
        label_array = []
        image_dir = self.config.get_value('IMAGES', 'dir')
        image_dir += (str(label) + '/')
        # print('image_dir',image_dir)
        for jpg in tqdm(jpgs):
            np_image, np_image_gray = self.read_img(image_dir + jpg)
            contour_measurement = self.get_contour_measurement(np_image_gray)
            segmented, labels, k = self.process_img(np_image)
            kmean_measurements = self.get_kmean_measurement(np_image, labels, k)
            is_center_valid = self.is_center_valid(kmean_measurements, contour_measurement)
            if is_center_valid:
                params = self.calculate_crop_size(np_image, kmean_measurements)
                if not params:
                    params = self.calculate_crop_size(np_image, contour_measurement)
            else:
                params = self.calculate_crop_size(np_image, contour_measurement)
            if params:
                np_image = self.crop_img(np_image, params)
                # if label in ['3', '11', '13', '19']:
                # 	plt.imshow(np_image)
                # 	plt.show()
                if np_image is not None:
                    if augment:
                        images_aug = self.augmentation(np_image)
                        for np_image in images_aug:
                            np_image = self.format_image(np_image)
                            image_array.append(np_image)
                            label_array.append(int(label))
                    else:
                        np_image = self.format_image(np_image)
                        image_array.append(np_image)
                        label_array.append(int(label))
            else:
                self.count['no params'] += 1
        print("total valid samples in class {}: {}".format(label, len(image_array)))
        return image_array, label_array

    def process_image(self, image_dir):
        np_image, np_image_gray = self.read_img(image_dir)
        contour_measurement = self.get_contour_measurement(np_image_gray)
        segmented, labels, k = self.process_img(np_image)
        kmean_measurements = self.get_kmean_measurement(np_image, labels, k)
        is_center_valid = self.is_center_valid(kmean_measurements, contour_measurement)
        if is_center_valid:
            params = self.calculate_crop_size(np_image, kmean_measurements)
            if not params:
                params = self.calculate_crop_size(np_image, contour_measurement)
        else:
            params = self.calculate_crop_size(np_image, contour_measurement)
        if params:
            converted_params = self.convert_crop_size(np_image, params)
            np_image = self.crop_img(np_image, params)
            if np_image is not None:
                np_image = self.format_image(np_image)
                return [np_image, converted_params]
        return None


if __name__ == '__main__':
    IP = ImageProcessor()
    image = IP.read_img("images/4.jpeg")
    segmented, labels = IP.process_img(image)
    measurements = IP.get_measurement(image, labels)
    params = IP.calculate_crop_size(image, measurements)
    IP.crop_img(image, params)

