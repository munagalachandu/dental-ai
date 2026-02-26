from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

app = FastAPI()

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    return {"segments_detected": int(len(instances))}
