from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from config import *
from measurement_utils import *
from implant_logic import *

app = FastAPI()

# Load model once
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    zoom: float = Form(DEFAULT_ZOOM)   # 👈 USER INPUT
):

    # If zoom invalid → fallback
    if zoom <= 0:
        zoom = DEFAULT_ZOOM

    PX_TO_MM = BASE_PIXEL_SIZE * zoom
    MM_TO_PX = 1 / PX_TO_MM

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    masks = instances.pred_masks.numpy()

    results = []

    for mask in masks:

        polygon = polygon_from_mask(mask)
        if polygon is None:
            continue

        height, top, bottom = calc_height(polygon)

        width_2 = calc_width(mask, top, bottom, MM_TO_PX, 2)
        width_6 = calc_width(mask, top, bottom, MM_TO_PX, 6)
        width_8 = calc_width(mask, top, bottom, MM_TO_PX, 8)

        height_mm = height * PX_TO_MM
        width_mm = width_2 * PX_TO_MM

        implant_w, implant_h = recommend_implant(height_mm, width_mm)

        results.append({
            "zoom_used": zoom,
            "height_mm": float(height_mm),
            "width_2mm": float(width_2 * PX_TO_MM),
            "width_6mm": float(width_6 * PX_TO_MM),
            "width_8mm": float(width_8 * PX_TO_MM),
            "implant_width": float(implant_w),
            "implant_height": float(implant_h)
        })

    return {"analysis": results}
