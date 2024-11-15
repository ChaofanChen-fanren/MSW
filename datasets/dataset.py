import os
import json
import random

import numpy as np
from PIL import Image
import torch.utils.data as data
from common import object_list


class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list = object_list[dataset_name]

    def __len__(self):
        return self.length

    # TODO: 增加数据增强 随机增加三张图像进行拼接
    # TODO: 实现链接：https://github.com/CASIA-IVA-Lab/FiLo/blob/main/datasets/mvtec_supervised.py#L44
    def combine_img(self, cls_name):
        img_paths = os.path.join(self.root, cls_name, 'test')
        img_ls = []
        mask_ls = []
        for i in range(4):
            defect = os.listdir(img_paths)
            random_defect = random.choice(defect)
            files = os.listdir(os.path.join(img_paths, random_defect))
            random_file = random.choice(files)
            img_path = os.path.join(img_paths, random_defect, random_file)
            mask_path = os.path.join(self.root, cls_name, 'ground_truth', random_defect, random_file[:3] + '_mask.png')
            img = Image.open(img_path)
            img_ls.append(img)
            if random_defect == 'good':
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            mask_ls.append(img_mask)
        # image
        image_width, image_height = img_ls[0].size
        result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
        for i, img in enumerate(img_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_image.paste(img, (x, y))

        # mask
        result_mask = Image.new("L", (2 * image_width, 2 * image_height))
        for i, img in enumerate(mask_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_mask.paste(img, (x, y))

        return result_image, result_mask

    # TODO: 这里batch_size=1, 因为这里class进行区分，而anomaly_clip进行区分
    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name.replace("_", " "), 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path)}


if __name__ == "__main__":
    dataset = Dataset(root="/Users/chenchaofan/python_project/data/VisA", transform=None, target_transform=None, dataset_name='visa')
    data = dataset[0]
    print(data)