from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.evaluation import CityscapesInstanceEvaluator, inference_on_dataset

from model import MaskRCNN
from collections import OrderedDict
import logging
import torch
import os

cityscapes = [\
	"cityscapes_fine_instance_seg_train",\
	"cityscapes_fine_sem_seg_train",\
	"cityscapes_fine_instance_seg_val",\
	"cityscapes_fine_sem_seg_val",\
	"cityscapes_fine_instance_seg_test",\
	"cityscapes_fine_sem_seg_test",\
	"cityscapes_fine_panoptic_train",\
	"cityscapes_fine_panoptic_val",\
	"cityscapes"\
]

#import sys
#if len(sys.argv) == 1:
#	raise ValueError("need dataset name")
#assert sys.argv[1] in cityscapes, "Wrong dataset name"
TRAIN_DATASET_NAME = "cityscapes_fine_instance_seg_val"
TEST_DATASET_NAME = "cityscapes_fine_instance_seg_val"

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_model(cls, cfg):
        return MaskRCNN(cfg)

    #@classmethod
    #def build_optimizer(cls, cfg):
    #    pass

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = DatasetCatalog.get(TRAIN_DATASET_NAME)
        mapper = DatasetMapper(cfg, is_train=True)
        data_loader = build_detection_train_loader(
	        dataset=dataset,
	        mapper=mapper,
	        sampler = None,
	        total_batch_size = 2,           # 2 images for 1 step
	        aspect_ratio_grouping=True,
	        num_workers=0,
	        collate_fn=None
        )

        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return CityscapesInstanceEvaluator(TEST_DATASET_NAME)

def trainer_cfg(cfg):
    #cfg.SOLVER.AMP.ENABLED = False # SimpleTrainer

    #optimizer
    #cfg.SOLVER.BASE_LR
    #cfg.SOLVER.WEIGHT_DECAY_NORM
    #cfg.SOLVER.BIAS_LR_FACTOR
    #cfg.SOLVER.WEIGHT_DECAY_BIAS
    #cfg.SOLVER.BASE_LR
    #cfg.SOLVER.NESTEROV
    #cfg.SOLVER.WEIGHT_DECAY

    #lr_scheduler
    #cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.STEPS = (6, 9)
    cfg.SOLVER.MAX_ITER = 20
    cfg.SOLVER.GAMMA = 0.1
    #cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    #cfg.SOLVER.WARMUP_ITERS = 1000
    #cfg.SOLVER.WARMUP_METHOD = "linear"

    return cfg
def mapper_cfg(cfg):
    cfg.SOLVER.IMS_PER_BATCH = 16
    #cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.DATALOADER.NUM_WORKERS = 4

    #cfg.INPUT.MIN_SIZE_TRAIN = (800, )
    cfg.INPUT.MAX_SIZE_TRAIN = 1024 #1333
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    #cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1024 #1333
    cfg.INPUT.RANDOM_FLIP = "none"       # choose one of ["horizontal, "vertical", "none"]

    #cfg.INPUT.CROP.ENABLED = False
    #cfg.INPUT.CROP.TYPE = "relative_range"
    #cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.MASK_ON = True #False
    #cfg.INPUT.MASK_FORMAT = "polygon"
    #cfg.MODEL.KEYPOINT_ON = False

    return cfg
def resnet_cfg(cfg):
    # RESNET
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    #cfg.MODEL.BACKBONE.FREEZE_AT = 2
    #cfg.MODEL.RESNETS.DEPTH = 50
    #cfg.MODEL.RESNETS.NUM_GROUPS = 1               # 1 ==> ResNet; > 1 ==> ResNeXt
    #cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64         # Baseline width of each group.
    #cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    #cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    #cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True         # Place the stride 2 conv on the 1x1 filter
    #cfg.MODEL.RESNETS.RES5_DILATION = 1            # Apply dilation in stage "res5"
    #cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    #cfg.MODEL.RESNETS.DEFORM_MODULATED = False
    #cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1

    return cfg
def fpn_cfg(cfg):
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    #cfg.MODEL.FPN.OUT_CHANNELS = 256
    #cfg.MODEL.FPN.NORM = ""
    #cfg.MODEL.FPN.FUSE_TYPE = "sum"  # Can be either "sum" or "avg"

    return cfg
def rpn_cfg(cfg):
    #Proposal_generator
    #cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
    #cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
    #cfg.MODEL.RPN.NMS_THRESH = 0.7
    #cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.33 #0.5
    #cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
    #cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    #cfg.MODEL.RPN.BOUNDARY_THRESH = -1          # Set to -1 to disable pruning anchors
    #cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    #cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
    #cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.0
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000 #12000
    #cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    #cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]
    #RPN head
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.CONV_DIMS = [512] # -1 : out_channels == in_channels
    #cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    #cfg.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    #cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0

    return cfg
def roi_cfg(cfg):
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads" #"Res5ROIHeads"

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # *per image* (number of regions of interest [ROIs])
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7 #80
    #cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
    #cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]     # Not Defined in original Mask RCNN
    #cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]

    #cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False

    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"] # ["res4"]
    #cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
    #cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    #cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead" # ""
    #cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
    #cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2 #0
    #cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
    #cfg.MODEL.ROI_BOX_HEAD.NORM = ""

    #cfg.MODEL.MASK_ON = True #False
    #cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    #cfg.VIS_PERIOD = 0
    #cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 4 #0
    #cfg.MODEL.ROI_MASK_HEAD.NORM = ""
    #cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False

    #cfg.MODEL.KEYPOINT_ON = False

    return cfg
    
logger = logging.getLogger("detectron2")
if __name__ == "__main__":
    cfg = get_cfg()
    cfg = trainer_cfg(cfg)
    cfg = mapper_cfg(cfg)
    cfg = resnet_cfg(cfg)
    cfg = fpn_cfg(cfg)
    cfg = rpn_cfg(cfg)
    cfg = roi_cfg(cfg)

    #trainer = Trainer(cfg)
    #trainer.resume_or_load(resume=True)
    #trainer.train()
    #exit(0)
    model = MaskRCNN(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    #print(do_test(cfg, model))
    DATASET_NAME = "cityscapes_fine_instance_seg_val"
    dataset = DatasetCatalog.get(DATASET_NAME)
    mapper = DatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=None,
        num_workers=0,
        collate_fn=None
    )

    inputs = None
    outputs = None
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            inputs = data
            outputs = model(data)
            break
    
    #out = outputs[0]
    #bbox = out.pred_boxes.tensor    # [N, 4]
    #scores = out.scores             # [N]
    #classes = out.pred_classes      # [N]
    #mask = out.pred_masks           # [N, 1, 28, 28]
    mask = outputs[0].pred_masks
    outputs[0].pred_masks = mask[:, 0, :, :]


    d = inputs[0]
    import numpy as np
    from PIL import Image
    from detectron2.utils.file_io import PathManager
    from detectron2.data.catalog import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
    from cityscapesscripts.helpers.labels import labels

    dirname = "cityscapes-data-vis"
    os.makedirs(dirname, exist_ok=True)

    #thing_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]
    #meta = Metadata().set(thing_classes=thing_classes)
    meta = MetadataCatalog.get(DATASET_NAME)

    img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
    visualizer = Visualizer(img, metadata=meta)
    vis = visualizer.draw_instance_predictions(outputs[0])
    fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
    vis.save(fpath)

"""def get_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    print("evaluator_type :", evaluator_type)

    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
def do_test(cfg, model):
    results = OrderedDict()
    
    DATASET_NAME = "cityscapes_fine_instance_seg_val"
    dataset = DatasetCatalog.get(DATASET_NAME)
    mapper = DatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=None,
        num_workers=0,
        collate_fn=None
    )
    evaluator = get_evaluator(
        cfg, DATASET_NAME, os.path.join(cfg.OUTPUT_DIR, "inference", DATASET_NAME)
    )
    results_i = inference_on_dataset(model, data_loader, evaluator)
    results[DATASET_NAME] = results_i
    if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(DATASET_NAME))
        print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results"""