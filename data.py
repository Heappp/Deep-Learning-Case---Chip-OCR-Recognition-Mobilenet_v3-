import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from typing import Sequence


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


class MDataset(Dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        MyRotateTransform([0, 180]),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    name2label = {'74LS00': 0, '74LS01': 1, '74LS02': 2, '74LS03': 3, '74LS05': 4, '74LS06': 5, '74LS09': 6, '74LS11': 7, '74LS112': 8, '74LS123': 9, '74LS14': 10, '74LS169': 11, '74LS175': 12, '74LS190': 13, '74LS191': 14, '74LS192': 15, '74LS193': 16, '74LS42': 17, '74LS47': 18, '74LS48': 19, '74LS83': 20, '74LS85': 21}
    label2name = {0: '74LS00', 1: '74LS01', 2: '74LS02', 3: '74LS03', 4: '74LS05', 5: '74LS06', 6: '74LS09', 7: '74LS11', 8: '74LS112', 9: '74LS123', 10: '74LS14', 11: '74LS169', 12: '74LS175', 13: '74LS190', 14: '74LS191', 15: '74LS192', 16: '74LS193', 17: '74LS42', 18: '74LS47', 19: '74LS48', 20: '74LS83', 21: '74LS85'}

    def __init__(self, url):
        self.data, self.label = [], []
        for folder in MDataset.name2label.keys():
            filenames = os.listdir(url + '/' + folder)
            for filename in filenames:
                image = cv2.imread(url + '/' + folder + '/' + filename, cv2.IMREAD_GRAYSCALE)
                image = MDataset.pretreatment(image)

                self.data.append(image)
                self.label.append(MDataset.name2label[folder])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return MDataset.transform(self.data[index]), self.label[index]

    @staticmethod
    def pretreatment(image):
        # 二值化
        image = cv2.medianBlur(image, ksize=3)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -2)

        # 将矩形部分旋转为正
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(max_contour)
        angle = rect[2] if rect[1][0] > rect[1][1] else rect[2] - 90

        M = cv2.getRotationMatrix2D(rect[0], angle, 1)
        image = cv2.warpAffine(image, M, (1280, 960))

        # 截取矩形部分
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)

        w, h = max(rect[1][1], rect[1][0]), min(rect[1][1], rect[1][0])
        x, y = int(rect[0][0] - w / 2), int(rect[0][1] - h / 2)
        image = image[y:y + int(h), x:x + int(w)]

        # 开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

        return image
