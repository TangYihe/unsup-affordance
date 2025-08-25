import numpy as np
import torch


def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)


def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output


def cal_nss(pred: np.ndarray, gt: np.ndarray, threshold=0.1) -> np.ndarray:
    # pred = pred / 255.0
    # gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / std
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, threshold)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)

    return nss


def compute_cls_acc(preds, label):
    pred = torch.max(preds, 1)[1]
    # label = torch.max(labels, 1)[1]
    num_correct = (pred == label).sum()
    return float(num_correct) / float(preds.size(0))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def updata(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.cnt += n
        if self.cnt == 0:
            self.avg = 1
        else:
            self.avg = self.sum / self.cnt