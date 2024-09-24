from py_distance_transforms import transform_cuda
from juliacall import Main as jl
import torch
from typing import Callable
from torch import nn

class HD_loss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, apply_nonlin: Callable = None, alpha=2.0):
        super(HD_loss, self).__init__()
        self.alpha = alpha
        self.apply_nonlin = apply_nonlin

    @torch.no_grad()
    def distance_field(self, img):

        field = torch.zeros_like(img, dtype=torch.float32, device=img.device)
        out_shape = field.shape
        for batch in range(out_shape[0]):
            for c in range(out_shape[1]):
                fg_mask = img[batch, c] 
                if fg_mask.any():
                    bg_mask = ~fg_mask
                    fg_dist = transform_cuda(fg_mask)**0.5
                    bg_dist = transform_cuda(bg_mask)**0.5
                    field[batch, c] = fg_dist + bg_dist
        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        if self.apply_nonlin is not None:
            pred = self.apply_nonlin(pred)
        with torch.no_grad():
            target = torch.nn.functional.one_hot(target.long(), pred.shape[1]).permute(0, 4, 1, 2, 3)[:,1:]    
            pred_dt = self.distance_field(torch.nn.functional.one_hot(torch.argmax(pred, dim=1).long(), pred.shape[1]).permute(0, 4, 1, 2, 3)[:,1:].to(dtype=torch.bool))
            target_dt = self.distance_field(target.to(dtype=torch.bool))
        pred_error = (pred[:,1:] - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()
        return loss

if __name__ == '__main__':
    torch.manual_seed(1)
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 2, 64, 64, 64))
    ref = torch.randint(0, 1, (2, 64, 64, 64))

    dl_old = HD_loss(apply_nonlin=softmax_helper_dim1, alpha = 2)
    res_old = dl_old(pred, ref)
    print(res_old)

