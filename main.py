import os
import logging
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personal-color-bot")

PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")

app = FastAPI()


# -----------------------------
# Seasonal logic
# -----------------------------
def detect_season(rgb: Tuple[int, int, int]) -> Dict[str, Any]:
    r, g, b = rgb

    warm = r > b
    light = (r + g + b) / 3 > 160

    if warm and light:
        return {
            "season": "Spring",
            "undertone": "Warm",
            "palette": ["Peach", "Coral", "Warm Green"],
        }
    if warm and not light:
        return {
            "season": "Autumn",
            "undertone": "Warm",
            "palette": ["Olive", "Mustard", "Brown"],
        }
    if not warm and light:
        return {
            "season": "Summer",
            "undertone": "Cool",
            "palette": ["Lavender", "Soft Blue", "Rose"],
        }
    return {
        "season": "Winter",
        "undertone": "Cool",
        "palette": ["Black", "White", "Emerald"],
    }


def analyze_skin(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Simple heuristic: sample center region as face-like region.
    Returns average skin RGB + HEX + seasonal color info.
    """
    h, w, _ = image_bgr.shape

    # center crop
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4
    region = image_bgr[y1:y2, x1:x2]

    if region.size == 0:
        raise ValueError("Skin region is empty (image too small?)")

    avg_bgr = cv2.mean(region)[:3]  # (b,g,r)
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


# -----------------------------
# Webhook verification
# -----------------------------
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)

    return PlainTextResponse("Verification failed", status_code=403)


# -----------------------------
# Receive messages
# -----------------------------
@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            message = event.get("message", {})

            if not sender_id or not message:
                continue

            # handle images
            for att in message.get("attachments", []):
                if att.get("type") == "image":
                    image_url = att.get("payload", {}).get("url")
                    if image_url:
                        # background processing so webhook returns fast
                        background_tasks.add_task(
                            handle_image, sender_id, image_url)

    return {"status": "ok"}


# -----------------------------
# Image handler
# -----------------------------
def handle_image(sender_id: str, image_url: str):
    try:
        # download image
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()

        nparr = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            send_message(
                sender_id, "Зураг уншиж чадсангүй. Өөр зураг явуулж үзээрэй 🙏")
            return

        result = analyze_skin(image)
        season = result["seasonal_color"]

        reply_text = (
            "🎨 Personal Color (Seasonal)\n\n"
            f"• Season: {season['season']}\n"
            f"• Undertone: {season['undertone']}\n"
            f"• Best colors: {', '.join(season['palette'])}\n"
            f"• Avg skin: {result['skin']['average_skin_hex']}"
        )

        send_message(sender_id, reply_text)

    except Exception as e:
        logger.exception("handle_image failed: %s", e)
        send_message(
            sender_id, "Алдаа гарлаа 😅 Дахиад нэг удаа зураг явуулж үзээрэй.")


# -----------------------------
# Send message back to user
# -----------------------------
def send_message(psid: str, text: str):
    if not PAGE_ACCESS_TOKEN:
        logger.error(
            "PAGE_ACCESS_TOKEN missing. Set it in Render Environment Variables.")
        return

    url = "https://graph.facebook.com/v18.0/me/messages"
    params = {"access_token": PAGE_ACCESS_TOKEN}
    payload = {"recipient": {"id": psid}, "message": {"text": text}}

    try:
        r = requests.post(url, params=params, json=payload, timeout=15)
        if r.status_code >= 400:
            logger.error("FB send_message error %s: %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("send_message failed: %s", e)


@app.get("/")
async def health():
    return {"ok": True}
