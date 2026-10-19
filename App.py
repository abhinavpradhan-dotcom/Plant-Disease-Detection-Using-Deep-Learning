from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "plant_disease_best.pt"
RECORD_FILE = BASE_DIR / "streamlit_prediction_records.csv"

FIELDNAMES = [
    "time", "image", "disease", "confidence",
    "confidence_percent", "x1", "y1", "x2", "y2"
]

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
)

def find_model_path() -> Path:
    env_path = os.environ.get("PLANT_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    discovered = list(BASE_DIR.rglob("*.pt"))
    if discovered:
        return discovered[0]
    raise FileNotFoundError(
        "Model not found. Put plant_disease_best.pt inside "
        "plant_disease_website/models/ or set PLANT_MODEL_PATH."
    )

@st.cache_resource
def load_model(model_file: str) -> YOLO:
    return YOLO(model_file)

def ensure_record_file() -> None:
    if not RECORD_FILE.exists():
        with RECORD_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()

def save_records(records: list[dict]) -> None:
    if not records:
        return
    ensure_record_file()
    with RECORD_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writerows(records)

def predict_image(model: YOLO, image: Image.Image, image_name: str, confidence: float) -> tuple[np.ndarray, list[dict]]:
    # Run prediction directly from memory (no temporary files on disk)
    results = model.predict(source=image, imgsz=416, conf=confidence, save=False)
    result = results[0]

    # result.plot() returns BGR numpy array
    annotated_array = result.plot()
    # Convert BGR to RGB for PIL / Streamlit display
    annotated_image = annotated_array[:, :, ::-1]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records: list[dict] = []

    if len(result.boxes) == 0:
        records.append({
            "time": now,
            "image": image_name,
            "disease": "No disease detected",
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "x1": "", "y1": "", "x2": "", "y2": "",
        })
    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            disease_name = model.names[class_id]
            x1, y1, x2, y2 = [round(val, 2) for val in box.xyxy[0].tolist()]
            records.append({
                "time": now,
                "image": image_name,
                "disease": disease_name,
                "confidence": round(score, 4),
                "confidence_percent": round(score * 100, 2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    save_records(records)
    return annotated_image, records

def read_records() -> pd.DataFrame:
    ensure_record_file()
    return pd.read_csv(RECORD_FILE)

# --- App Layout ---
st.title("Plant Disease Detection")
st.caption("YOLO powered leaf scanning app for upload and camera prediction.")

try:
    model_path = find_model_path()
    model = load_model(str(model_path))
    st.success(f"Model loaded: {model_path.name}")
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.write("Classes:", len(model.names))
    
    records_df = read_records()
    st.download_button(
        "Download prediction records",
        data=records_df.to_csv(index=False),
        file_name="plant_disease_prediction_records.csv",
        mime="text/csv",
    )

tab_upload, tab_camera, tab_records = st.tabs(["Upload Images", "Camera", "Records"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload plant leaf images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            source_image = Image.open(uploaded_file).convert("RGB")
            annotated_image, records = predict_image(
                model=model,
                image=source_image,
                image_name=uploaded_file.name,
                confidence=confidence,
            )

            st.subheader(uploaded_file.name)
            col_orig, col_res = st.columns(2)
            with col_orig:
                st.image(source_image, caption="Original image", use_container_width=True)
            with col_res:
                st.image(annotated_image, caption="Prediction result", use_container_width=True)
            st.dataframe(pd.DataFrame(records), use_container_width=True)

with tab_camera:
    camera_file = st.camera_input("Capture a plant leaf image")
    if camera_file is not None:
        camera_image = Image.open(camera_file).convert("RGB")
        annotated_image, records = predict_image(
            model=model,
            image=camera_image,
            image_name="camera_leaf.jpg",
            confidence=confidence,
        )

        col_orig, col_res = st.columns(2)
        with col_orig:
            st.image(camera_image, caption="Captured image", use_container_width=True)
        with col_res:
            st.image(annotated_image, caption="Prediction result", use_container_width=True)
        st.dataframe(pd.DataFrame(records), use_container_width=True)

with tab_records:
    st.subheader("Prediction history")
    history_df = read_records()
    if not history_df.empty:
        st.dataframe(history_df.tail(100).iloc[::-1], use_container_width=True)
    else:
        st.info("No records logged yet.")