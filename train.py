
"""train the Ultra-Fast-Lane-Detection model"""
try:
    from moxing.framework import file
    print("import moxing success")
except ModuleNotFoundError as e:
    print(f'not modelarts env, error={e}')

import os
import time
import json
import random

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.train.callback._time_monitor import TimeMonitor
from mindspore.train.callback._loss_monitor import LossMonitor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.communication import init
from mindspore.train.model import ParallelMode
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback._callback import Callback

from src.resnet import get_resnet
from src.network import ParsingNet
from src.utils import print_trainable_params_count, TusimpleAccEval
from src.dataset import create_lane_train_dataset, create_lane_test_dataset
from src.config import config as cfg
from src.loss import TrainLoss, NetWithLossCell
from src.lr_scheduler import warmup_cosine_annealing_lr_V2

#from mindspore.profiler import Profiler

ms.common.set_seed(12345)
random.seed(12345)
np.random.seed(12345)


class Val_Callback(Callback):
    """
    Valid the test data at every epoch.

    """

    def __init__(self, model, val_dataset, device_id, label_info_list, output_path='./output'):
        super(Val_Callback, self).__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.device_id = device_id
        self.label_info_list = label_info_list
        if output_path.startswith('s3://') or output_path.startswith('obs://'):
            self.output_path = '/cache/output'
            self.obs_output_path = output_path
        else:
            self.output_path = output_path
            self.obs_output_path = None
        self.best_acc = 0

        self.accEval = TusimpleAccEval()

    def epoch_end(self, run_context):
        """
        Call at each epoch end.
        """
        print("============== Starting Testing ==============")
        start = time.time()
        acc = 0
        total_count = 0
        for data in self.val_dataset.create_dict_iterator():
            imgs = data['image']
            batch_index = data['index'].asnumpy()
            results = self.model.predict(imgs).asnumpy()
            for i in range(results.shape[0]):
                index = batch_index[i]
                gt_lanes = np.array(self.label_info_list[index]['lanes'])
                y_samples = np.array(self.label_info_list[index]['h_samples'])

                pred_one_img_lanes = self.accEval.generate_tusimple_lines(
                    results[i], imgs[0, 0].shape, cfg.griding_num)
                one_img_acc = self.accEval.bench(pred_one_img_lanes,
                                                 gt_lanes, y_samples)
                acc += one_img_acc
                total_count += 1
                if total_count % 100 == 0:
                    print(f'Eval count:{total_count}')
        acc = acc / total_count
        end = time.time()
        use_time = end - start
        print(
            f"============== accuracy:{acc}, use_time:{use_time} ==============")

        if acc > self.best_acc:
            self.best_acc = acc
            if acc >= 0.9582:
                print('=================save checkpoint....====================')
                cb_params = run_context.original_args()
                epoch = cb_params.cur_epoch_num
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)
                file_name = f'model_epoch{epoch}_id{self.device_id}_{self.best_acc}.ckpt'
                ckpt_file = os.path.join(self.output_path, file_name)
                save_checkpoint(cb_params.train_network, ckpt_file)
                if self.obs_output_path is not None:
                    file.copy_parallel(ckpt_file, os.path.join(
                        self.obs_output_path, file_name))
                print('==============save checkpoint finished===================')
        print(f'The best acc is {self.best_acc}')


def main():
    device_id = int(os.getenv('DEVICE_ID', 0))
    device_num = int(os.getenv('RANK_SIZE', 1))
    print(f'device_id:{device_id}')
    print(f'device_num:{device_num}')
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target)
    context.set_context(device_id=device_id)
#    context.set_context(enable_graph_kernel=True)

    #profiler = Profiler()

    if cfg.start_epochs < 1 or cfg.start_epochs > cfg.epochs:
        print(f'start_epochs must between 1 and {cfg.epochs}')
        return

    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    if cfg.resume.startswith('s3://') or cfg.resume.startswith('obs://'):
        local_resume = os.path.join(
            cfg.local_data_root, f'resume_{device_id}.ckpt')
        file.copy_parallel(cfg.resume, local_resume)
        cfg.resume = local_resume

    if cfg.backbone_pretrain.startswith('s3://') or cfg.backbone_pretrain.startswith('obs://'):
        local_backbone_pretrain = os.path.join(
            cfg.local_data_root, f'backbone_pretrain_{device_id}.ckpt')
        file.copy_parallel(cfg.backbone_pretrain, local_backbone_pretrain)
        cfg.backbone_pretrain = local_backbone_pretrain

    backbone = get_resnet(resnet_type=cfg.backbone)
    if cfg.backbone_pretrain and cfg.backbone_pretrain != 'None':
        param_dict = load_checkpoint(cfg.backbone_pretrain)
        load_param_into_net(backbone, param_dict)
        print('load resnet pretrain ckpt success')

    net = ParsingNet(cfg.backbone, backbone, cls_dim=(
        cfg.griding_num + 1, len(cfg.row_anchor), cfg.num_lanes), use_aux=True)

    print_trainable_params_count(net)

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
        train_root = os.path.join(local_data_path, 'train_set')
        train_gt_list = 'train_gt.txt'
    elif dataset == 'CULane':
        train_root = local_data_path
        train_gt_list = 'list/train_gt.txt'

    train_dataset = create_lane_train_dataset(train_root, train_gt_list, cfg.batch_size,
                                              rank_size=device_num, rank_id=device_id)
    if dataset == 'Tusimple':
        val_dataset = create_lane_test_dataset(dataset,
                                               os.path.join(local_data_path, 'test_set'), 'test_label.json', 1)

        with open(os.path.join(local_data_path, 'test_set', 'test_label.json')) as f:
            label_lines = f.readlines()
        label_info_list = []
        for i in range(len(label_lines)):
            json_data = json.loads(label_lines[i])
            label_info_list.append(json_data)

    batches_per_epoch = train_dataset.get_dataset_size()
    print(f'batches_per_epoch:{batches_per_epoch}')

    data_type = ms.float32 if cfg.amp_level == 'O0' else ms.float16
    loss_fn = TrainLoss(data_type=data_type)
    net_with_loss = NetWithLossCell(net, loss_fn)

    if cfg.use_cosine_decay_lr == 'True':
        lr = warmup_cosine_annealing_lr_V2(lr=cfg.lr, steps_per_epoch=batches_per_epoch, warmup_epochs=cfg.warmup,
                                           warmup_init_lr=cfg.warmup_init_lr,
                                           max_epoch=cfg.epochs - cfg.cooldown, T_max=cfg.epochs, eta_min=cfg.lr_min,
                                           cooldown_epochs=cfg.cooldown)
    else:
        lr = []
        for epoch_index in range(cfg.epochs):
            for _ in range(batches_per_epoch):
                step_lr = (1.0 - epoch_index / cfg.epochs) * cfg.lr
                lr.append(step_lr)
        lr = np.array(lr).astype(np.float32)
    cfg.start_epochs = cfg.start_epochs - 1
    if cfg.start_epochs > 0:
        lr = lr[cfg.start_epochs * batches_per_epoch:]
    lr = Tensor(lr)

    if cfg.optimizer == 'Adam':
        opt = nn.Adam(
            net.trainable_params(),
            lr,
            weight_decay=cfg.weight_decay,
            use_nesterov=True,
            loss_scale=1024.0 if cfg.amp_level == 'O3' else 1.0
        )
    else:
        opt = nn.SGD(
            net.trainable_params(),
            lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
            loss_scale=1024.0 if cfg.amp_level == 'O3' else 1.0
        )

    loss_scale_manager = FixedLossScaleManager(
        1024.0, drop_overflow_update=False)
    if cfg.amp_level == 'O3':
        model = Model(net_with_loss, None, opt, boost_level='O0',
                      amp_level=cfg.amp_level, loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net_with_loss, None, opt, boost_level='O0',
                      amp_level=cfg.amp_level)

    loss_cb = LossMonitor(per_print_times=100)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    val_cb = None
    if dataset == 'Tusimple':
        val_cb = Val_Callback(
            model, val_dataset, device_id, label_info_list, cfg.train_url)

    callbacks = [time_cb, loss_cb]

    if val_cb is not None:
        callbacks += [val_cb]

    cfg.dataset_sink_mode = True if cfg.dataset_sink_mode == 'True' else False
    model.train(cfg.epochs - cfg.start_epochs, train_dataset,
                callbacks=callbacks, dataset_sink_mode=cfg.dataset_sink_mode)

    # profiler.analyse()

    if dataset == 'CULane':
        if cfg.train_url.startswith('s3://') or cfg.train_url.startswith('obs://'):
            output_path = f'/cache/output_{device_id}'
            obs_output_path = cfg.train_url
        else:
            output_path = cfg.train_url
            obs_output_path = None

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        file_name = f'model_culane_{device_id}.ckpt'
        ckpt_file = os.path.join(output_path, file_name)
        save_checkpoint(net, ckpt_file)
        if obs_output_path is not None:
            file.copy_parallel(ckpt_file, os.path.join(
                obs_output_path, file_name))
        print('==============save checkpoint finished===================')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f'Total time: {duration}s.', duration)
