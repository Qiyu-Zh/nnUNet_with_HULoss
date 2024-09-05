from py_distance_transforms import transform_cuda
from juliacall import Main as jl
import torch
from typing import Callable
from torch import nn


def compute_dtm_gpu(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    # Convert img_gt to float if not already float
    if img_gt.dtype != torch.float32:
        img_gt = img_gt.float()

    fg_dtm = torch.zeros(out_shape, dtype=torch.float32, device=img_gt.device)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b, c]
            if posmask.bool().any():
                posdis = transform_cuda(posmask)
                fg_dtm[b, c] = posdis

    return fg_dtm.to(img_gt.dtype)  


class HD_loss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, power = 2):
        super(HD_loss, self).__init__()
        self.power = power
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        num_classes = x.shape[1]
        if len(x.shape) == 5:
            y = torch.nn.functional.one_hot(y.long(), num_classes).permute(0, 4, 1, 2, 3)
            x = torch.nn.functional.one_hot(torch.argmax(x, dim=1).long(), num_classes).permute(0, 4, 1, 2, 3)
        else:
            y = torch.nn.functional.one_hot(y.long(), num_classes).permute(0, 3, 1, 2)
            x = torch.nn.functional.one_hot(torch.argmax(x, dim=1).long(), num_classes).permute(0, 3, 1, 2)
        gt_dtm = compute_dtm_gpu(y, x.shape) ** self.power
        # print(gt_dtm.shape)
        delta_s = (x - y.float()) ** self.power
        seg_dtm = compute_dtm_gpu(x, x.shape) ** self.power
        dtm = gt_dtm + seg_dtm

        multipled = torch.einsum('bcxyz, bcxyz->bcxyz', delta_s, dtm)
        return multipled[:,1:].mean()
   


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 10, 32, 32, 32))
    ref = torch.randint(0, 9, (2, 32, 32, 32))

    dl_old = HD_loss(apply_nonlin=softmax_helper_dim1, power = 4)
    res_old = dl_old(pred, ref)
    print(res_old)
