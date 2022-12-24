
"""eval the Ultra-Fast-Lane-Detection model"""
try:
    from moxing.framework import file
    print("import moxing success")
except ModuleNotFoundError as e:
    print(f'not modelarts env, error={e}')

import os
import time
import json

import numpy as np

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import get_resnet
from src.network import ParsingNet
from src.utils import print_trainable_params_count, TusimpleAccEval, CULaneF1Eval
from src.dataset import create_lane_test_dataset
from src.config import config as cfg


def main():
    device_id = int(os.getenv('DEVICE_ID', 0))
    device_num = int(os.getenv('RANK_SIZE', 1))
    print(f'device_id:{device_id}')
    print(f'device_num:{device_num}')
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target)
    context.set_context(device_id=device_id)
#    context.set_context(enable_graph_kernel=True)

    backbone = get_resnet(resnet_type=cfg.backbone)
    net = ParsingNet(cfg.backbone, backbone, cls_dim=(
        cfg.griding_num + 1, len(cfg.row_anchor), cfg.num_lanes), use_aux=False)
    print_trainable_params_count(net)

    if cfg.resume.startswith('s3://') or cfg.resume.startswith('obs://'):
        local_resume = os.path.join(
            cfg.local_data_root, f'resume_{device_id}.ckpt')
        file.copy_parallel(cfg.resume, local_resume)
        cfg.resume = local_resume

    if cfg.resume and cfg.resume != 'None':
        ckpt = load_checkpoint(cfg.resume)
        load_param_into_net(net, ckpt)
        print('load ckpt success')

    copy_result_file = os.path.join(
        cfg.local_data_root, 'local_data_path.txt')
    if device_num == 1 or device_id == 0:
        if cfg.data_url.startswith('s3://') or cfg.data_url.startswith('obs://'):
            start = time.time()
            print('start copy data...')
            local_data_path = os.path.join(cfg.local_data_root, 'data')
            file.copy_parallel(cfg.data_url, local_data_path)
            end = time.time()
            print(f'copy data finished,use time{end-start}s')
        else:
            local_data_path = cfg.data_url

        with open(copy_result_file, 'w') as f:
            f.write(local_data_path)

    if device_num > 1 and device_id != 0:
        local_data_path = None
        while local_data_path is None:
            time.sleep(5)
            if os.path.exists(copy_result_file):
                with open(copy_result_file) as f:
                    local_data_path = f.readline()

    dataset = cfg.dataset

    if dataset == 'Tusimple':
        with open(os.path.join(local_data_path, 'test_set', 'test_label.json')) as f:
            label_lines = f.readlines()
        label_info_list = []
        for i in range(len(label_lines)):
            json_data = json.loads(label_lines[i])
            label_info_list.append(json_data)

        val_dataset = create_lane_test_dataset(dataset,
                                               os.path.join(local_data_path, 'test_set'), 'test_label.json', cfg.batch_size)

        accEval = TusimpleAccEval()
        acc = 0
        total_count = 0
        for data in val_dataset.create_dict_iterator():
            imgs = data['image']
            batch_index = data['index'].asnumpy()
            results = net(imgs).asnumpy()
            for i in range(results.shape[0]):
                index = batch_index[i]
                gt_lanes = np.array(label_info_list[index]['lanes'])
                y_samples = np.array(label_info_list[index]['h_samples'])

                pred_one_img_lanes = accEval.generate_tusimple_lines(
                    results[i], imgs[0, 0].shape, cfg.griding_num)
                one_img_acc = accEval.bench(pred_one_img_lanes,
                                            gt_lanes, y_samples)
                acc += one_img_acc
                total_count += 1
                if total_count % 100 == 0:
                    print(f'total_count:{total_count}')
        acc = acc / total_count
        print(f'accuracy:{acc}')

    elif dataset == 'CULane':

        culane_list_path = os.path.join(local_data_path, 'list/test.txt')
        with open(culane_list_path) as f:
            list_img = f.readlines()
        all_anno_lanes = []
        for img_file in list_img:
            img_file = img_file.strip()
            if img_file[0] == '/':
                img_file = img_file[1:]
            label_file = img_file.replace('.jpg', '.lines.txt')
            label_path = os.path.join(local_data_path, label_file)
            with open(label_path) as f:
                label_lines = f.readlines()

            anno_lanes = []
            for line in label_lines:
                cur_lane = []
                line_info = line.split()
                for i in range(int(len(line_info) / 2)):
                    cur_lane.append(
                        (float(line_info[i * 2]), float(line_info[i * 2 + 1])))
                anno_lanes.append(cur_lane)

            all_anno_lanes.append(anno_lanes)

        val_dataset = create_lane_test_dataset(dataset,
                                               local_data_path, 'list/test.txt', cfg.batch_size)
        f1_eval = CULaneF1Eval()
        total_count = 0
        for data in val_dataset.create_dict_iterator():
            imgs = data['image']
            batch_index = data['index'].asnumpy()
            results = net(imgs).asnumpy()
            for i in range(len(results)):
                cls_out = results[i]
                index = batch_index[i]
                anno_lanes = all_anno_lanes[index]
                f1_eval.cal_tp_fp_fn(anno_lanes, cls_out,
                                     imgs[0, 0].shape, cfg.griding_num)
                total_count += 1
                print(f'Eval count:{total_count}')
        f1 = f1_eval.get_f1()
        print(f'f1:{f1}')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f'Total time: {duration}s.', duration)
