import math
import torch
from torch import nn


class IOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type
    
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        
        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)
        
        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)
        
        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect
        
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError
        
        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


class CenternessIOULoss(nn.Module):
    def __init__(self, loc_loss_type='centerness_iou'):
        super(CenternessIOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type
    
    def forward(self, reg_pred, reg_target, ctrness_pred, ctrness_targets, weight=None):
        """
        Args:
            reg_pred: Nx4 predicted bounding boxes
            reg_target: Nx4 target bounding boxes
            ctrness_pred : Nx1 predicted bounding boxes
            ctrness_targets : Nx1 predicted bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = reg_pred[:, 0]
        pred_top = reg_pred[:, 1]
        pred_right = reg_pred[:, 2]
        pred_bottom = reg_pred[:, 3]
        
        target_left = reg_target[:, 0]
        target_top = reg_target[:, 1]
        target_right = reg_target[:, 2]
        target_bottom = reg_target[:, 3]
        
        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)
        
        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)
        
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        
        # cIoU 计算两个中心点之间的距离
        center_x1 = ctrness_pred[:, 0]
        center_x2 = ctrness_targets[:, 0]
        
        center_y1 = ctrness_pred[:, 1]
        center_y2 = ctrness_targets[:, 1]
        
        inter_diag = (center_x1 - center_x2) ** 2 + (
                center_y1 - center_y2) ** 2
        
        c_diag = torch.clamp((torch.max(pred_right, target_right) - torch.min(pred_left, target_left)),
                             min=0) ** 2 \
                 + torch.clamp(
            (torch.max(pred_bottom, target_bottom) - torch.min(pred_top, target_top)), min=0) ** 2
        u = (inter_diag) / c_diag
        
        # cIoU 计算两个bboxing的宽高比
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            (target_right - center_x2) / (target_bottom - center_y2)) - torch.atan(
            (pred_right - center_x1) / (pred_bottom - center_y1))), 2) \
            + (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            (target_right - center_x2) / (target_bottom - center_y2)) - torch.atan(
            (pred_right - center_x1) / (pred_bottom - center_y1))), 2)
        
        with torch.no_grad():
            S = (ious > 0.5).float()
            alpha = S * v / (1 - ious + v)
        
        cious = ious - u - alpha * v
        cious = torch.clamp(cious, min=-1.0, max=1.0)
        
        losses = 1 - cious
        
        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()
