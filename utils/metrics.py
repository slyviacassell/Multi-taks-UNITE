# coding=utf-8

'''
Created: 2021/2/1
@author: Slyviacassell@github.com
'''

import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np


# Build Losses
def normal_loss(prediction: torch.Tensor, gt: torch.Tensor, l2_normalization: bool = False, ignore_label=255):
    '''Compute normal loss. (normalized cosine distance)
      :param: prediction: the output of cnn. Tensor type
      :param: gt: the groundtruth. Tensor type
      :param: l2_normalization: whether to perform l2 normalization for prediction and gt
      :param: ignore_label: ignored label. Int type
    '''
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    # mask the pixel whose all channels are same with ignore label
    mask = (torch.sum(torch.abs(gt - ignore_label) < 1e-12, dim=1) - 3).nonzero(as_tuple=False).squeeze()
    prediction = prediction[mask]
    gt = gt[mask]
    # unitization
    if l2_normalization:
        prediction = F.normalize(prediction, dim=1)
        gt = F.normalize(gt, dim=1)

    loss = F.cosine_similarity(gt, prediction)
    return 1 - loss.mean()


# references form astmt
class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(nn.Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """

    def __init__(self, size_average=True, normalize=False, norm=1):

        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad

        if ignore_label:
            n_valid = torch.sum(torch.abs(label - ignore_label) > 1e-12).item()
            out[torch.abs(label - ignore_label) < 1e-12] = 0
            label[torch.abs(label - ignore_label) < 1e-12] = 0

        if self.normalize is not None:
            out = self.normalize(out)
            # costume
            label = self.normalize(label)

        loss = self.loss_func(out, label, reduction='sum')

        if self.size_average:
            if ignore_label:
                loss.div_(max(n_valid, 1e-6))
            else:
                loss.div_(float(np.prod(label.size())))

        return loss


def pascal_normal_loss():
    return NormalsLoss(normalize=True)


# https://github.com/CoinCheung/pytorch-loss/blob/d76a6f8eaaab6fcf0cc89c011e1917159711c9fb/label_smooth.py#L14
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        :param: logits: tensor of shape (N, C, H, W)
        :param: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


def get_conf_mat(seg_pred: torch.Tensor, seg_gt: torch.Tensor, n_classes: int, ignore_index=255):
    gt = seg_gt.squeeze(1).view(-1)
    pred = seg_pred
    ids = gt != ignore_index

    logits = torch.argmax(pred, dim=1).view(-1)
    logits = logits[ids]
    gt = gt[ids]
    # mat = confusion_matrix(gt.cpu().numpy(), logits.cpu().numpy(), np.arange(n_classes))
    mat = confusion_matrix_2(gt, logits, n_classes)
    return mat


def get_pixel_acc(seg_pred: torch.Tensor, seg_gt: torch.Tensor, ignore_index=255):
    gt = seg_gt.squeeze(1).view(-1)
    pred = seg_pred

    ids = gt != ignore_index

    logits = torch.argmax(pred, dim=1).view(-1)
    logits = logits[ids]
    gt = gt[ids]
    pixel_acc = gt == logits
    return pixel_acc


def get_mIoU(mat: np.ndarray, n_classes: int=None):
    # use cycle
    # jaccard_perclass = []
    # for i in range(n_classes):
    #     if not mat[i, i] == 0:
    #         jaccard_perclass.append(mat[i, i] / (np.sum(mat[i, :]) + np.sum(mat[:, i]) - mat[i, i]))
    # mIoU = np.sum(jaccard_perclass) / len(jaccard_perclass)

    # not use cycle
    eps = 1e-12
    IoU = (np.diag(mat) + eps) / (np.sum(mat, axis=0) + np.sum(mat, axis=1) - np.diag(mat) + eps)
    mIoU = np.mean(IoU)
    return mIoU


def get_normal_cosine(prediction: torch.Tensor, gt: torch.Tensor, normalization: bool = False,
                      ignore_label=255):
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    # not compatible for distributed
    # mask = ((gt == ignore_label).sum(dim=1) - 3).nonzero(as_tuple=False).squeeze()
    # mask = ((torch.abs(gt - ignore_label) < 1e-12).sum(dim=1) - 3).nonzero(
    #     as_tuple=False).squeeze()
    # prediction = prediction[mask]
    # gt = gt[mask]
    mask = ((torch.abs(gt - ignore_label) < 1e-12).sum(dim=1) - 3) != 0
    gt[torch.logical_not(mask), :] = 0.
    prediction[torch.logical_not(mask), :] = 0.
    if normalization:
        prediction = F.normalize(prediction, dim=1)
        gt = F.normalize(gt, dim=1)
    # cosine similarity implements normalization, but you can still normalize input
    cosine = F.cosine_similarity(gt, prediction)
    return cosine, mask


def get_normal_cosine_pascal(prediction: torch.Tensor, gt: torch.Tensor):
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    mask = torch.abs(torch.linalg.norm(gt, ord=2, dim=1)) > 1e-12
    mask = torch.logical_and(((torch.abs(gt - 255) < 1e-12).sum(dim=1) - 3) != 0, mask)
    prediction[torch.logical_not(mask), :] = 0.
    gt[torch.logical_not(mask), :] = 0.
    prediction = F.normalize(prediction, dim=1)
    gt = F.normalize(gt, dim=1)
    # cosine similarity implements normalization, but you can still normalize input
    cosine = F.cosine_similarity(gt, prediction)
    return cosine, mask


# can only be used at the end of the test epoch
def get_normal_metrics(cosine_list: list) -> dict:
    metrics = {}
    overall_cos = np.clip(np.concatenate(cosine_list), -1, 1)
    # angles = np.arccos(overall_cos) / np.pi * 180.0
    angles = np.rad2deg(np.arccos(overall_cos))
    metrics['mean'] = np.mean(angles)
    metrics['rmse'] = np.sqrt(np.mean(angles ** 2))
    metrics['median'] = np.median(angles)
    metrics['11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
    metrics['22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
    metrics['30'] = np.mean(np.less_equal(angles, 30.0)) * 100
    metrics['45'] = np.mean(np.less_equal(angles, 45.0)) * 100
    return metrics


# this requires more GPU Mem and can be used for running statistics
def get_normal_metrics_torch(cosine: torch.Tensor, mask: torch.Tensor):
    metrics = {}
    overall_cos = torch.clip(cosine, -1, 1)
    angles = torch.rad2deg(torch.arccos(overall_cos))
    angles_2 = angles ** 2
    metrics['angles_sum'] = torch.sum(angles[mask], dtype=torch.float64)
    metrics['angles_2_sum'] = torch.sum(angles_2[mask], dtype=torch.float64)
    metrics['11.25'] = torch.sum(torch.less_equal(angles[mask], 11.25).float())
    metrics['22.5'] = torch.sum(torch.less_equal(angles[mask], 22.5).float())
    metrics['30'] = torch.sum(torch.less_equal(angles[mask], 30.0).float())
    metrics['45'] = torch.sum(torch.less_equal(angles[mask], 45.0).float())
    return metrics, angles


# rewrite sklearn method to torch
def confusion_matrix_1(y_true: torch.Tensor, y_pred: torch.Tensor, N=None):
    if N is None:
        N = max(torch.max(y_true)[0], torch.max(y_pred)[0]) + 1
    y_true = y_true.long()
    y_pred = y_pred.long()
    return torch.sparse.LongTensor(
        torch.stack([y_true, y_pred]),
        torch.ones_like(y_true, dtype=torch.long),
        torch.Size([N, N])).to_dense()


# weird trick with bincount
def confusion_matrix_2(y_true, y_pred, N=None):
    if N is None:
        N = max(max(y_true), max(y_pred)) + 1
    y_true = y_true.long()
    y_pred = y_pred.long()
    y = N * y_true + y_pred
    y = torch.bincount(y, minlength=N ** 2).view(N, N)
    # if len(y) < N * N:
    #     y = torch.cat((y, torch.zeros(N * N - len(y), dtype=torch.long)))
    # y = y.reshape(N, N)
    return y
