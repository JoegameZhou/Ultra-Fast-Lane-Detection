
import os
import random

import json
import numpy as np
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision

from src.config import config


class LaneTrainDataset:
    def __init__(self, root_path, list_path):

        self.root_path = root_path

        with open(list_path, 'r') as f:
            self.list = f.readlines()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.root_path, label_name)
        label = Image.open(label_path)

        img_path = os.path.join(self.root_path, img_name)
        img = Image.open(img_path)

        return img, label

    def __len__(self):
        return len(self.list)


class LaneTestDataset:
    def __init__(self, dataset, root_path, test_label_file):

        self.dataset = dataset
        self.root_path = root_path

        with open(test_label_file, 'r') as f:
            self.list = f.readlines()

    def __getitem__(self, index):
        if self.dataset == 'Tusimple':
            json_line = self.list[index]
            json_data = json.loads(json_line)
            raw_file = json_data['raw_file']
            if raw_file[0] == '/':
                raw_file = raw_file[1:]

            img_path = os.path.join(self.root_path, raw_file)
            img = Image.open(img_path)

            return img, index
        elif self.dataset == 'CULane':
            img_file = self.list[index]
            img_file = img_file.strip()
            if img_file[0] == '/':
                img_file = img_file[1:]

            img_path = os.path.join(self.root_path, img_file)
            img = Image.open(img_path)

            return img, index

    def __len__(self):
        return len(self.list)


def random_process(img, label, angle=6, max_offset1=100, max_offset2=200):
    angle = random.randint(0, angle * 2) - angle
    offset1 = np.random.randint(-max_offset1, max_offset1)
    offset2 = np.random.randint(-max_offset2, max_offset2)

    # image rotate
    img = Image.fromarray(img)
    img = img.rotate(angle, resample=Image.BILINEAR)
    # iamge UDoffsetLABEL
    w, h = img.size
    img = np.array(img)
    if offset1 > 0:
        img[offset1:, :, :] = img[0:h - offset1, :, :]
        img[:offset1, :, :] = 0
    if offset1 < 0:
        real_offset = -offset1
        img[0:h - real_offset, :, :] = img[real_offset:, :, :]
        img[h - real_offset:, :, :] = 0
    # image RandomLROffsetLABEL
    if offset2 > 0:
        img[:, offset2:, :] = img[:, 0:w - offset2, :]
        img[:, :offset2, :] = 0
    if offset2 < 0:
        real_offset = -offset2
        img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
        img[:, w - real_offset:, :] = 0

    # label rotate
    label = Image.fromarray(label)
    label = label.rotate(angle, resample=Image.NEAREST)
    # label UDoffsetLABEL
    label = np.array(label)
    if offset1 > 0:
        label[offset1:, :] = label[0:h - offset1, :]
        label[:offset1, :] = 0
    if offset1 < 0:
        offset1 = -offset1
        label[0:h - offset1, :] = label[offset1:, :]
        label[h - offset1:, :] = 0
    # label RandomLROffsetLABEL
    if offset2 > 0:
        label[:, offset2:] = label[:, 0:w - offset2]
        label[:, :offset2] = 0
    if offset2 < 0:
        offset2 = -offset2
        label[:, 0:w - offset2] = label[:, offset2:]
        label[:, w - offset2:] = 0

    return img, label


def find_start_pos(row_sample, start_line):
    l, r = 0, len(row_sample) - 1
    while True:
        mid = int((l + r) / 2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid


def get_cls_label(label):
    h = label.shape[0]
    w = label.shape[1]

    def scale_f(x): return int((x * 1.0 / 288) * h)
    sample_tmp = list(map(scale_f, config.row_anchor))

    all_idx = np.zeros((config.num_lanes, len(sample_tmp), 2))
    for i, r in enumerate(sample_tmp):
        label_r = label[int(round(r))]
        for lane_idx in range(1, config.num_lanes + 1):
            pos = np.where(label_r == lane_idx)[0]
            if len(pos) == 0:
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = -1
                continue
            pos = np.mean(pos)
            all_idx[lane_idx - 1, i, 0] = r
            all_idx[lane_idx - 1, i, 1] = pos

    all_idx_cp = all_idx.copy()
    for i in range(config.num_lanes):
        if np.all(all_idx_cp[i, :, 1] == -1):
            continue
        # if there is no lane

        valid = all_idx_cp[i, :, 1] != -1
        valid_idx = all_idx_cp[i, valid, :]
        if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
            continue
        if len(valid_idx) < 6:
            continue

        valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
        p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
        start_line = valid_idx_half[-1, 0]
        pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

        fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
        fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

        assert np.all(all_idx_cp[i, pos:, 1] == -1)
        all_idx_cp[i, pos:, 1] = fitted

    num_lane, n, n2 = all_idx_cp.shape
    col_sample = np.linspace(0, w - 1, config.griding_num)

    cls_label = np.zeros((n, num_lane))
    for i in range(num_lane):
        pti = all_idx_cp[i, :, 1]
        cls_label[:, i] = np.asarray(
            [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else config.griding_num for pt in pti])
    cls_label = cls_label.astype(np.int32)
    return label, cls_label


def free_scale_mask(label, size=(36, 100)):
    label = Image.fromarray(label)
    seg_label = label.resize((size[1], size[0]), Image.NEAREST)
    seg_label = np.array(seg_label).astype(np.int32)
    return seg_label


def create_lane_train_dataset(data_root_path, data_list_path, batch_size,
                              num_workers=8, rank_size=1, rank_id=0):

    ds.config.set_seed(1234)
    ds.config.set_num_parallel_workers(num_workers)

    lane_dataset = LaneTrainDataset(
        data_root_path, os.path.join(data_root_path, data_list_path))

    dataset = ds.GeneratorDataset(source=lane_dataset, column_names=['image', 'label'],
                                  #                                  num_parallel_workers=num_workers,
                                  shuffle=True, num_shards=rank_size, shard_id=rank_id)

    dataset = dataset.map(operations=[random_process],
                          input_columns=['image', 'label'],
                          output_columns=['image', 'label'],
                          num_parallel_workers=num_workers)

    dataset = dataset.map(operations=[get_cls_label],
                          input_columns=['label'],
                          output_columns=['label', 'cls_label'], column_order=['image', 'cls_label', 'label'],
                          num_parallel_workers=num_workers)

    dataset = dataset.map(operations=[free_scale_mask],
                          input_columns=['label'],
                          output_columns=['seg_label'],
                          num_parallel_workers=num_workers)

    transform_img = [
        vision.Resize((288, 800)),
        vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                         std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        vision.HWC2CHW()
    ]
    dataset = dataset.map(operations=transform_img,
                          input_columns=['image'],
                          output_columns=['image'],
                          num_parallel_workers=num_workers)

    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return dataset


def create_lane_test_dataset(dataset, data_root_path, test_label_file, batch_size,
                             num_workers=8, rank_size=1, rank_id=0):

    ds.config.set_num_parallel_workers(num_workers)

    lane_dataset = LaneTestDataset(dataset,
                                   data_root_path, os.path.join(data_root_path, test_label_file))

    dataset = ds.GeneratorDataset(source=lane_dataset, column_names=['image', 'index'],
                                  shuffle=False, num_shards=rank_size, shard_id=rank_id)

    transform_img = [
        vision.Resize((288, 800)),
        vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                         std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        vision.HWC2CHW()
    ]
    dataset = dataset.map(operations=transform_img,
                          input_columns=['image'],
                          output_columns=['image'],
                          num_parallel_workers=num_workers)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return dataset


if __name__ == "__main__":
    dataset = create_lane_test_dataset('CULane',
                                       '../../../dataset/CULane/', 'list/test.txt', 1, num_workers=8)
    data = next(dataset.create_dict_iterator())
