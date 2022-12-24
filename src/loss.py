

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor

from src.config import config as cfg


class SoftmaxFocalLoss(nn.Cell):
    def __init__(self, gamma=2):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.softmax = P.Softmax(axis=1)
        self.pow = P.Pow()
#        self.log_softmax = nn.LogSoftmax(axis=1)
        self.softmax2 = P.Softmax(axis=1)
        self.log = P.Log()

        self.griding_num = cfg.griding_num
        self.weight = Tensor(
            np.ones((self.griding_num + 1,))).astype(np.float32)
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.nll = P.NLLLoss(reduction="mean")

    def construct(self, logits, labels):
        scores = self.softmax(logits)
        factor = self.pow(1.0 - scores, self.gamma)

        logits2 = self.softmax2(logits)
        log_score = self.log(logits2)

        log_score = factor * log_score
        log_score = self.transpose(log_score, (0, 2, 3, 1))
        log_score = self.reshape(log_score, (-1, self.griding_num + 1))
        labels = self.reshape(labels, (-1,))
        loss, _ = self.nll(log_score, labels, self.weight)
        return loss


class ParsingRelationLoss(nn.Cell):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
        self.concat = P.Concat(axis=0)
        self.zeros_like = P.ZerosLike()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=1.0)
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        loss = self.concat(loss_all)
        smooth_loss = self.smooth_l1_loss(loss, self.zeros_like(loss))
        return self.mean(smooth_loss)


class ParsingRelationDis(nn.Cell):
    def __init__(self, griding_num=100, anchor_nums=56, data_type=ms.float16):
        super(ParsingRelationDis, self).__init__()
        self.dim = griding_num
        self.num_rows = anchor_nums

        self.softmax = P.Softmax(axis=1)
        self.embedding = Tensor(
            np.arange(griding_num)).astype(data_type).view((1, -1, 1, 1))
        self.reduce_sum = P.ReduceSum(keep_dims=False)

        self.l1_loss = nn.L1Loss(reduction='mean')

    def construct(self, x):
        x = self.softmax(x[:, :self.dim, :, :])
        pos = self.reduce_sum(x * self.embedding, 1)
        diff_list1 = []
        for i in range(0, self.num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1_loss(diff_list1[i], diff_list1[i + 1])
        loss = loss / (len(diff_list1) - 1)
        return loss


class FocalLoss(nn.Cell):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.depth = cfg.griding_num + 1
        self.onehot = P.OneHot(axis=1)
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.softmax = P.Softmax(axis=1)
        self.focal_loss = nn.FocalLoss(
            weight=None, gamma=2.0, reduction='mean')

    def construct(self, cls_out, cls_label):
        cls_out = self.softmax(cls_out)
        cls_label = self.onehot(cls_label, self.depth,
                                self.on_value, self.off_value)
        return self.focal_loss(cls_out, cls_label)


class TrainLoss(nn.Cell):
    def __init__(self, gamma=2, data_type=ms.float16):
        super(TrainLoss, self).__init__()
        self.num_lanes = cfg.num_lanes
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

        self.w1 = 1.0
        self.loss1 = SoftmaxFocalLoss(gamma=gamma)
#        self.loss1 = FocalLoss(weight=None, gamma=2.0, reduction='mean')
        self.w2 = cfg.sim_loss_w
        self.loss2 = ParsingRelationLoss()
        self.w3 = 1.0
        self.loss3 = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='mean')
        self.w4 = cfg.shp_loss_w
        self.loss4 = ParsingRelationDis(
            griding_num=cfg.griding_num, anchor_nums=len(cfg.row_anchor), data_type=data_type)

    def construct(self, cls_out, seg_out, cls_label, seg_label):
        total_loss = self.w1 * self.loss1(cls_out, cls_label)
        if self.w2 > 0:
            total_loss += self.w2 * self.loss2(cls_out)
        total_loss += self.w3 * self.loss3(self.reshape(self.transpose(
            seg_out, (0, 2, 3, 1)), (-1, self.num_lanes + 1)), self.reshape(seg_label, (-1,)))
        if self.w4 > 0:
            total_loss += self.w4 * self.loss4(cls_out)
        return total_loss


class NetWithLossCell(nn.Cell):
    def __init__(self, network, loss_fn):
        super(NetWithLossCell, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, x, cls_label=None, seg_label=None):
        if self.training:
            cls_out, seg_out = self.network(x)
            loss = self.loss_fn(cls_out, seg_out, cls_label, seg_label)
            return loss
        else:
            cls_out = self.network(x)
            return cls_out
