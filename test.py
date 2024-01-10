import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from maskdino.config import add_maskformer2_config

# #####################################3
WEIGHTS_PATH = "../swindocsegmenter/model_final_prima_swindocseg.pth"
CONFIG_PATH = "./config_doclay.yaml"
# Class names in the proper order
# PRIMA config and weights:
# CLASSES = np.asarray(
#     [
#         "Background",
#         "TextRegion",
#         "ImageRegion",
#         "TableRegion",
#         "MathsRegion",
#         "SeparatorRegion",
#         "OtherRegion",
#     ]
# )
# OR
# DocLayNet config and weights:
CLASSES = np.asarray(
    [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    ]
)
# #####################################3

# Color for each class
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (127, 0, 0),
    (0, 127, 0),
    (0, 0, 127),
    (127, 127, 127),
]


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


def draw_box(img, box, color, label):
    # box: [x0,y0,x1,y1]
    box = [round(x) for x in box]
    color = COLORS[color]

    # draw a bounding box rectangle and label on the image
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    text = "{}: {:.4f}".format(CLASSES[label], 1.0)
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


# Load model and weights
cfg = setup({"config_file": CONFIG_PATH, "opts": []})
model = build_model(cfg)  # returns a torch.nn.module
DetectionCheckpointer(model).load(WEIGHTS_PATH)

img1 = cv2.imread("./example0.jpg")
img2 = cv2.imread("./example1.png")


def preprocess_image(img: np.ndarray):
    # resize for faster inference and RAM saving
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    # img = cv2.resize(img, (1024, 1024))
    # bgr to rgb
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hwc to chw format
    img_ = torch.tensor(img_).permute(2, 0, 1)
    # float32 format
    img_ = img_.float().cuda()
    return img_


# inference
model.eval()
with torch.no_grad():
    for j, img in enumerate([img1, img2]):
        img_ = preprocess_image(img)
        outputs = model(
            [{"image": img_, "height": img_.shape[1], "width": img_.shape[2]}]
        )

        # Draw boxes on image
        for output in outputs:
            pred_classes = output["instances"].pred_classes.cpu().numpy()
            boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
            scores = output["instances"].scores.cpu().numpy()

            # filter out low confidence boxes
            boxes = boxes[scores > 0.5]
            pred_classes = pred_classes[scores > 0.5]

            # draw boxes on image
            for i, box in enumerate(boxes):
                box *= 2
                img = draw_box(img, box, pred_classes[i], pred_classes[i])

        # save image
        cv2.imwrite(f"example_segmented{j}.png", img)
