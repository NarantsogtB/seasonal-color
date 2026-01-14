import os
import logging
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personal-color-analyzer")

app = FastAPI()


# ---------- core logic ----------
def detect_season(rgb: Tuple[int, int, int]) -> Dict[str, Any]:
    r, g, b = rgb
    warm = r > b
    light = (r + g + b) / 3 > 160

    if warm and light:
        return {"season": "Spring", "undertone": "Warm", "palette": ["Peach", "Coral", "Warm Green"]}
    if warm and not light:
        return {"season": "Autumn", "undertone": "Warm", "palette": ["Olive", "Mustard", "Brown"]}
    if not warm and light:
        return {"season": "Summer", "undertone": "Cool", "palette": ["Lavender", "Soft Blue", "Rose"]}
    return {"season": "Winter", "undertone": "Cool", "palette": ["Black", "White", "Emerald"]}


def analyze_skin(image_bgr: np.ndarray) -> Dict[str, Any]:
    h, w, _ = image_bgr.shape
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4
    region = image_bgr[y1:y2, x1:x2]
    if region.size == 0:
        raise ValueError("Skin region is empty")

    avg_bgr = cv2.mean(region)[:3]
    avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
    avg_hex = "#{:02x}{:02x}{:02x}".format(*avg_rgb)

    season_info = detect_season(avg_rgb)
    return {
        "skin": {
            "average_skin_rgb": avg_rgb,
            "average_skin_hex": avg_hex,
        },
        "seasonal_color": season_info,
    }


# ---------- API ----------
class AnalyzeRequest(BaseModel):
    image_url: str


@app.get("/")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        # Download image
        r = requests.get(req.image_url, timeout=15)
        r.raise_for_status()

        nparr = np.frombuffer(r.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Cannot decode image")

        result = analyze_skin(image)
        return {"ok": True, "result": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze failed: %s", e)
        raise HTTPException(status_code=500, detail="Analyze failed")
