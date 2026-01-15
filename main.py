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

    logger.info(
        "🔬 detect_season: r=%d g=%d b=%d warm=%s light=%s",
        r, g, b, warm, light
    )

    if warm and light:
        return {"season": "Spring", "undertone": "Warm", "palette": ["Peach", "Coral", "Warm Green"]}
    if warm and not light:
        return {"season": "Autumn", "undertone": "Warm", "palette": ["Olive", "Mustard", "Brown"]}
    if not warm and light:
        return {"season": "Summer", "undertone": "Cool", "palette": ["Lavender", "Soft Blue", "Rose"]}
    return {"season": "Winter", "undertone": "Cool", "palette": ["Black", "White", "Emerald"]}


def analyze_skin(image_bgr: np.ndarray):
    h, w, _ = image_bgr.shape

    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4

    region = image_bgr[y1:y2, x1:x2]

    logger.info(
        "🟩 Skin region crop: x1=%d y1=%d x2=%d y2=%d",
        x1, y1, x2, y2
    )

    avg_bgr = cv2.mean(region)[:3]
    avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))

    logger.info("📊 avg_bgr=%s avg_rgb=%s", avg_bgr, avg_rgb)

    season_info = detect_season(avg_rgb)

    logger.info(
        "🧭 detect_season -> season=%s undertone=%s",
        season_info["season"],
        season_info["undertone"]
    )

    return {
        "skin": {
            "average_skin_rgb": avg_rgb,
            "average_skin_hex": "#{:02x}{:02x}{:02x}".format(*avg_rgb),
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
        logger.info("📥 Incoming image_url=%s", req.image_url)

        r = requests.get(req.image_url, timeout=15)
        r.raise_for_status()

        nparr = np.frombuffer(r.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Cannot decode image")

        h, w, _ = image.shape
        logger.info("🖼️ Image shape: %sx%s", w, h)

        result = analyze_skin(image)

        avg_rgb = result["skin"]["average_skin_rgb"]
        brightness = sum(avg_rgb) / 3
        season = result["seasonal_color"]["season"]
        undertone = result["seasonal_color"]["undertone"]

        logger.info(
            "🎨 avg_rgb=%s brightness=%.1f season=%s undertone=%s",
            avg_rgb,
            brightness,
            season,
            undertone,
        )

        return {"ok": True, "result": result}

    except Exception as e:
        logger.exception("❌ Analyze failed")
        raise HTTPException(status_code=500, detail="Analyze failed")
