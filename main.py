import logging
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personal-color-analyzer")

app = FastAPI()

mp_face_detection = mp.solutions.face_detection


# --------------------------
# Utilities
# --------------------------
def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance to reduce WB variance."""
    img = bgr.astype(np.float32)
    mean_b, mean_g, mean_r = img[:, :, 0].mean(
    ), img[:, :, 1].mean(), img[:, :, 2].mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    scale_b = mean_gray / (mean_b + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_r = mean_gray / (mean_r + 1e-6)
    img[:, :, 0] *= scale_b
    img[:, :, 1] *= scale_g
    img[:, :, 2] *= scale_r
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def download_image_bgr(image_url: str) -> np.ndarray:
    r = requests.get(image_url, timeout=15)
    r.raise_for_status()
    nparr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    return img


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def rect_from_rel_bbox(rel_bbox, w: int, h: int) -> Tuple[int, int, int, int]:
    """Return pixel bbox (x1,y1,x2,y2) from mediapipe relative bbox."""
    x1 = int(rel_bbox.xmin * w)
    y1 = int(rel_bbox.ymin * h)
    bw = int(rel_bbox.width * w)
    bh = int(rel_bbox.height * h)
    x2 = x1 + bw
    y2 = y1 + bh
    # clamp
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 1, w)
    y2 = clamp(y2, 1, h)
    return x1, y1, x2, y2


def skin_samples_from_face_bbox(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Take cheek + forehead regions inside face bbox.
    Returns concatenated pixels (Nx3).
    """
    x1, y1, x2, y2 = bbox
    face = bgr[y1:y2, x1:x2]
    if face.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    fh, fw, _ = face.shape

    # Define simple regions (relative)
    # Forehead region: top-center
    fx1, fy1 = int(0.30 * fw), int(0.10 * fh)
    fx2, fy2 = int(0.70 * fw), int(0.30 * fh)

    # Left cheek region
    lx1, ly1 = int(0.15 * fw), int(0.45 * fh)
    lx2, ly2 = int(0.35 * fw), int(0.70 * fh)

    # Right cheek region
    rx1, ry1 = int(0.65 * fw), int(0.45 * fh)
    rx2, ry2 = int(0.85 * fw), int(0.70 * fh)

    forehead = face[fy1:fy2, fx1:fx2]
    lcheek = face[ly1:ly2, lx1:lx2]
    rcheek = face[ry1:ry2, rx1:rx2]

    parts = [forehead, lcheek, rcheek]
    pixels = []
    for p in parts:
        if p.size:
            # reshape to Nx3
            pixels.append(p.reshape(-1, 3))
    if not pixels:
        return np.empty((0, 3), dtype=np.uint8)

    all_px = np.vstack(pixels)

    return all_px


def filter_skin_like_pixels(bgr_pixels: np.ndarray) -> np.ndarray:
    """
    Rough skin filter in YCrCb to reduce background/hair influence.
    """
    if bgr_pixels.size == 0:
        return bgr_pixels

    # convert Nx3 to 1xN image for cv2 conversion
    img = bgr_pixels.reshape(1, -1, 3)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    Y, Cr, Cb = ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2]

    # common approximate bounds for skin in YCrCb
    mask = (Cr >= 135) & (Cr <= 180) & (Cb >= 85) & (Cb <= 135) & (Y >= 40)
    filtered = bgr_pixels[mask]
    # if filter removes too much, fall back
    if filtered.shape[0] < 500:
        return bgr_pixels
    return filtered


def bgr_to_lab_mean(bgr_pixels: np.ndarray) -> Tuple[float, float, float]:
    img = bgr_pixels.reshape(1, -1, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,
                                                       3).astype(np.float32)
    L = float(np.mean(lab[:, 0]))  # 0..255
    a = float(np.mean(lab[:, 1]))  # 0..255 (128 is neutral)
    b = float(np.mean(lab[:, 2]))  # 0..255 (128 is neutral)
    return L, a, b


def lab_to_undertone(L: float, a: float, b: float) -> str:
    # shift around neutral center 128
    b_shift = b - 128.0  # + => yellow-ish (warm), - => blue-ish (cool)
    return "Warm" if b_shift >= 3.0 else "Cool"  # small threshold for stability


def lab_features(L: float, a: float, b: float) -> Dict[str, float]:
    a_shift = a - 128.0
    b_shift = b - 128.0
    chroma = float(np.sqrt(a_shift * a_shift + b_shift * b_shift))
    # normalize L from 0..255 to 0..100-ish for readability
    L_norm = (L / 255.0) * 100.0
    return {"L": L_norm, "a_shift": float(a_shift), "b_shift": float(b_shift), "chroma": chroma}


def season_from_features(feat: Dict[str, float]) -> Dict[str, Any]:
    """
    Stable rule-based mapping:
    - Undertone from b_shift
    - Lightness from L
    - Chroma from chroma
    """
    L = feat["L"]
    b_shift = feat["b_shift"]
    chroma = feat["chroma"]

    undertone = "Warm" if b_shift >= 3.0 else "Cool"

    # Lightness thresholds (tuned for common phone images)
    is_light = L >= 62.0
    # Chroma thresholds (how vivid)
    is_vivid = chroma >= 18.0

    if undertone == "Warm" and is_light:
        season = "Spring"
        palette = ["Peach", "Coral", "Warm Green", "Cream", "Camel"]
    elif undertone == "Warm" and not is_light:
        season = "Autumn"
        palette = ["Olive", "Mustard", "Brown", "Rust", "Terracotta"]
    elif undertone == "Cool" and is_light:
        season = "Summer"
        palette = ["Lavender", "Soft Blue", "Rose", "Powder Pink", "Cool Gray"]
    else:
        season = "Winter"
        palette = ["Black", "White", "Emerald", "Cobalt", "Fuchsia"]

    return {"season": season, "undertone": undertone, "palette": palette, "is_light": is_light, "is_vivid": is_vivid}


def confidence_score(face_found: bool, skin_px_count: int, feat: Dict[str, float]) -> float:
    """
    Simple confidence:
    - face found
    - enough skin pixels
    - not too dark / too bright
    """
    score = 0.0
    if face_found:
        score += 0.45
    score += min(0.35, skin_px_count / 8000.0 * 0.35)

    L = feat["L"]
    if 40.0 <= L <= 85.0:
        score += 0.20
    else:
        score += 0.05

    return float(min(1.0, score))


def analyze_one_image(image_url: str) -> Dict[str, Any]:
    if not image_url or not image_url.startswith("http"):
        raise HTTPException(status_code=400, detail="image_url is required")

    logger.info("📥 Incoming image_url=%s", image_url)
    bgr = download_image_bgr(image_url)
    h, w, _ = bgr.shape
    logger.info("🖼️ Image shape: %sx%s", w, h)

    bgr = gray_world_white_balance(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    face_found = False
    bbox = None

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        res = detector.process(rgb)
        if res.detections:
            # take best detection
            det = max(res.detections, key=lambda d: d.score[0])
            bbox = rect_from_rel_bbox(
                det.location_data.relative_bounding_box, w, h)
            face_found = True

    if not face_found or bbox is None:
        # fallback: center crop (still return, but low confidence)
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4
        logger.info("⚠️ Face not found. Using fallback crop.")
        region = bgr[y1:y2, x1:x2].reshape(-1, 3)
        region = filter_skin_like_pixels(region)
        L, a, b = bgr_to_lab_mean(region)
        feat = lab_features(L, a, b)
        season_info = season_from_features(feat)
        conf = confidence_score(False, region.shape[0], feat)
        return {
            "image_url": image_url,
            "face_found": False,
            "bbox": None,
            "lab_features": feat,
            "seasonal_color": season_info,
            "confidence": conf,
        }

    x1, y1, x2, y2 = bbox
    logger.info("🙂 Face bbox: x1=%d y1=%d x2=%d y2=%d", x1, y1, x2, y2)

    px = skin_samples_from_face_bbox(bgr, bbox)
    px = filter_skin_like_pixels(px)

    L, a, b = bgr_to_lab_mean(px)
    feat = lab_features(L, a, b)
    season_info = season_from_features(feat)
    conf = confidence_score(True, px.shape[0], feat)

    logger.info("🎨 feat=%s season=%s undertone=%s conf=%.2f", feat,
                season_info["season"], season_info["undertone"], conf)

    return {
        "image_url": image_url,
        "face_found": True,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "lab_features": feat,
        "seasonal_color": season_info,
        "confidence": conf,
        "skin_px_count": int(px.shape[0]),
    }


def median_features(results: List[Dict[str, Any]]) -> Dict[str, float]:
    Ls = [r["lab_features"]["L"] for r in results]
    aS = [r["lab_features"]["a_shift"] for r in results]
    bS = [r["lab_features"]["b_shift"] for r in results]
    ch = [r["lab_features"]["chroma"] for r in results]
    return {
        "L": float(np.median(Ls)),
        "a_shift": float(np.median(aS)),
        "b_shift": float(np.median(bS)),
        "chroma": float(np.median(ch)),
    }


# --------------------------
# API Models
# --------------------------
class AnalyzeRequest(BaseModel):
    image_url: str


class AnalyzeBatchRequest(BaseModel):
    image_urls: List[str]


@app.get("/")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = analyze_one_image(req.image_url)

    # warning when low confidence
    warning = None
    if result["confidence"] < 0.55:
        warning = (
            "Зурагны гэрэл/сүүдэр эсвэл нүүр илрүүлэлтээс шалтгаалан нарийвчлал буурч магадгүй. "
            "Цонхны байгалийн гэрэлд, нүүрээ төвд нь, filterгүй, 1x камераар дахин явуулбал илүү зөв гарна."
        )

    return {"ok": True, "result": result, "warning": warning}


@app.post("/analyze_batch")
def analyze_batch(req: AnalyzeBatchRequest):
    if not req.image_urls or len(req.image_urls) < 3:
        raise HTTPException(status_code=400, detail="Provide 3 image_urls")

    # take first 3
    urls = req.image_urls[:3]
    per_image = [analyze_one_image(u) for u in urls]

    med = median_features(per_image)
    combined = season_from_features(med)

    # confidence = median of confs (simple)
    conf = float(np.median([r["confidence"] for r in per_image]))

    warning = None
    if conf < 0.60:
        warning = (
            "Зургийн гэрэлтүүлэг, камерын тохиргоо, нүүрний сүүдрээс шалтгаалан алдаа гарч болно. "
            "Цонхны гэрэлд, нүүр төвд, filterгүй 3 зураг явуулбал илүү тогтвортой болно."
        )

    return {
        "ok": True,
        "result": {
            "combined_features": med,
            "seasonal_color": combined,
            "confidence": conf,
            "per_image": per_image,
        },
        "warning": warning,
    }
