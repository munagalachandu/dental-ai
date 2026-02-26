from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

app = FastAPI()

BASE_PIXEL_SIZE = 0.0825
DEFAULT_ZOOM = 0.78

# ================= LOAD MODEL =================
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

# ================= HELPERS =================

def polygon_from_mask(mask):
    contours, _ = cv2.findContours(mask.astype("uint8"),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def calc_height(polygon):
    pts = polygon.astype(np.float32)
    center = np.mean(pts, axis=0)
    pts_centered = pts - center

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    projections = pts_centered @ axis
    min_proj, max_proj = np.min(projections), np.max(projections)

    p1 = center + axis * min_proj
    p2 = center + axis * max_proj

    top, bottom = (p1, p2) if p1[1] < p2[1] else (p2, p1)
    return np.linalg.norm(bottom - top), top, bottom


def calc_width(mask, top, bottom, mm_to_px, offset_mm):

    h_vec = bottom - top
    h_unit = h_vec / np.linalg.norm(h_vec)

    offset_px = offset_mm * mm_to_px
    point = top + h_unit * offset_px
    cx, cy = int(point[0]), int(point[1])

    p_vec = np.array([-h_unit[1], h_unit[0]])

    pts = []
    for t in np.linspace(-500, 500, 2000):
        x = int(cx + p_vec[0] * t)
        y = int(cy + p_vec[1] * t)
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if mask[y, x] == 1:
                pts.append((x, y))

    if len(pts) < 2:
        return 0

    left = np.array(pts[0])
    right = np.array(pts[-1])
    return np.linalg.norm(right - left)


def recommend_implant(height_mm, width_mm):
    return max(width_mm - 3, 0), height_mm

# ================= ROUTES =================

@app.get("/")
def health():
    return {"status": "running"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    zoom: float = Form(DEFAULT_ZOOM)
):

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
