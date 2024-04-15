from torch.utils.data import Dataset
import torch
import json, os, random, time
import cv2
import numpy as np
from PIL import Image

class ImageNetLT(Dataset):
    def __init__(self, data_root, mode="train", transform=None, equal_sampling=False):
        self.mode = mode
        self.transform = transform
        self.data_root = data_root

        if self.mode == "train":
            print("Loading train data ...")
            self.json_path = "./datasets/ImageNet_LT/train.json"
        elif self.mode == "val":
            print("Loading valid data ...")
            self.json_path = "./datasets/ImageNet_LT/test.json"

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]
        self.data = self.all_info["annotations"]
        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))

        self.class_dict = self._get_class_dict()

        self.equal_sampling = equal_sampling

        self.class_index2shot, self.shot2_class_index = self._class_split()

    def __getitem__(self, index):
        if self.equal_sampling and self.mode == 'train':
            sample_class = random.randint(0, self.num_classes - 1)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else 0
        )  # 0-index

        if self.mode != 'train':
            return image, image_label, self.class_index2shot[int(image_label)]
        else:
            return image, image_label

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        fpath = os.path.join(self.data_root, now_info["fpath"])
        with open(fpath, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def _class_split(self):
        # 0: Many-shot (100, inf); 1: Medium-shot [20, 100]; 2: Few-shot [0, 20)
        class_index2shot = {}
        shot2_class_index = {0: [], 1: [], 2: []}
        with open("./datasets/ImageNet_LT/train_class_number.json", "r") as f:
            class_number = json.load(f)
        for k, v in class_number.items():
            if v > 100:
                class_index2shot[int(k)] = 0
                shot2_class_index[0].append(int(k))
            elif v <= 100 and v >= 20:
                class_index2shot[int(k)] = 1
                shot2_class_index[1].append(int(k))
            elif v < 20:
                class_index2shot[int(k)] = 2
                shot2_class_index[2].append(int(k))
        return class_index2shot, shot2_class_index
