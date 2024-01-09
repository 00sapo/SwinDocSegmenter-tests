import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from maskdino.config import add_maskformer2_config


def setup(args):
    """
    create configs and perform basic setups.
    """

    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    cfg.freeze()
    default_setup(cfg, args)
    # shutup detectron2 completely
    # logger = setup_logger(
    #     output=cfg.output_dir,
    #     distributed_rank=comm.get_rank(),
    #     name="maskdino",
    #     configure_stdout=false,
    # )
    # logger.setlevel("error")
    return cfg


cfg = setup({"config_file": "./config_prima.yaml", "opts": []})
model = build_model(cfg)  # returns a torch.nn.module

DetectionCheckpointer(model).load(
    "../swindocsegmenter/model_final_prima_swindocseg.pth"
)  # load a file, usually from cfg.model.weights

img = cv2.imread("../LaudareAugmentation/image_tests/laudare/Cortona1.png")
# resize for faster inference and RAM saving
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
# img = cv2.resize(img, (1024, 1024))
# bgr to rgb
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# hwc to chw format
img_ = torch.tensor(img_).permute(2, 0, 1)
# float32 format
img_ = img_.float()

classes = np.asarray(
    [
        "background",
        "textregion",
        "imageregion",
        "tableregion",
        "mathsregion",
        "separatorregion",
        "otherregion",
    ]
)

model.eval()
with torch.no_grad():
    outputs = model([{"image": img_, "height": img_.shape[1], "width": img_.shape[2]}])


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
]


def draw_box(img, box, color, label):
    # box: [x0,y0,x1,y1]
    box = [round(x) for x in box]
    color = COLORS[color]

    # draw a bounding box rectangle and label on the image
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    text = "{}: {:.4f}".format(classes[label], 1.0)
    cv2.putText(
        img,
        text,
        (box[0], box[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

    return img


for output in outputs:
    pred_classes = output["instances"].pred_classes.cpu().numpy()
    boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
    # normalize boxes so that min-max is 0-1
    min_x = boxes[:, 0].min()
    min_y = boxes[:, 1].min()
    max_x = boxes[:, 2].max()
    max_y = boxes[:, 3].max()
    boxes[:, 0] = (boxes[:, 0] - min_x) / (max_x - min_x)
    boxes[:, 1] = (boxes[:, 1] - min_y) / (max_y - min_y)
    boxes[:, 2] = (boxes[:, 2] - min_x) / (max_x - min_x)
    boxes[:, 3] = (boxes[:, 3] - min_y) / (max_y - min_y)
    # denormalize boxes to original image size
    boxes[:, 0] = boxes[:, 0] * img.shape[1]
    boxes[:, 1] = boxes[:, 1] * img.shape[0]
    boxes[:, 2] = boxes[:, 2] * img.shape[1]
    boxes[:, 3] = boxes[:, 3] * img.shape[0]

    # draw boxes on image
    for i, box in enumerate(boxes):
        img = draw_box(img, box, pred_classes[i], pred_classes[i])

    cv2.imwrite("test.png", img)
