from detectron2.structures import ImageList #, Boxes, Instances
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.roi_heads.roi_heads import build_roi_heads

import torch
from torch import nn

class MaskRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.input_shape = ShapeSpec(channels=3, height=512, width=1024)
        self.backbone = build_resnet_fpn_backbone(cfg, self.input_shape)
        
        self.rpn_input_shape = self.backbone.output_shape()
        self.rpn_proposallayer = build_proposal_generator(cfg, self.rpn_input_shape)

        self.roi_input_shape = self.backbone.output_shape()
        del self.roi_input_shape["p6"]
        self.roi_heads = build_roi_heads(cfg, self.roi_input_shape)

    def forward(self, data):
        image_size = []     
        images = []
        instances = []

        for d in data:
            image_size.append( tuple( [ d['height'], d['width'] ] ) )
            images.append(d['image'])
            if self.training:
                instances.append(d['instances'])

        images = torch.stack(images, dim=0).float()     # Should Normalize ?
        gt_instances = instances if self.training else None

        feature_maps = self.backbone(images)

        images = ImageList(images, image_size)
        proposals, rpn_losses = self.rpn_proposallayer(
            images=images,
            features=feature_maps,
            gt_instances=gt_instances
        )

        del feature_maps["p6"]
        pred_instances, mrcnn_losses = self.roi_heads(
            images=images,
            features=feature_maps,
            proposals=proposals,
            targets=gt_instances
        )

        if self.training:
            losses = dict(
                rpn_losses.items() |
                mrcnn_losses.items()
            )
            print(sorted(losses.items()))
            return losses
        else:
            return pred_instances