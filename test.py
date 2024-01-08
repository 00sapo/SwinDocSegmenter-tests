import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino.config import add_maskformer2_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino"
    )
    return cfg


cfg = setup({"config_file": "./config_prima.yaml", "opts": []})
model = build_model(cfg)  # returns a torch.nn.Module

DetectionCheckpointer(model).load(
    "../swindocsegmenter/model_final_prima_swindocseg.pth"
)  # load a file, usually from cfg.MODEL.WEIGHTS

checkpointer = DetectionCheckpointer(model, save_dir="output")
# checkpointer.save("model_999")  # save to output/model_999.pth
__import__("ipdb").set_trace()
model.eval()
# with torch.no_grad():
#     outputs = model(inputs)
