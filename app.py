import os
import time
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

st.set_page_config(
    page_title="Road Damage Detection and Localization",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .header-box {
        background-color: #1B4332;
        color: #FFFFFF;
        padding: 20px 24px;
        border-radius: 8px;
        border-left: 6px solid #52B788;
        margin-bottom: 20px;
    }
    .header-box h2 {
        color: #FFFFFF !important;
        margin: 0 0 6px 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .header-box p {
        color: #F4F7F5 !important;
        margin: 0;
        font-size: 0.95rem;
    }
    .stButton>button {
        background-color: #2D6A4F;
        color: white;
        border-radius: 6px;
        border: 1px solid #1B4332;
    }
    .stButton>button:hover {
        background-color: #40916C;
        color: white;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def resolve_model_path() -> Optional[Path]:
    env_path = os.environ.get("MODEL_PATH", "").strip()
    if env_path:
        cand = Path(env_path)
        if cand.exists() and cand.is_file():
            return cand.resolve()

    local_default = Path.cwd() / "best_road_damage_model.pt"
    if local_default.exists() and local_default.is_file():
        return local_default.resolve()

    return None

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path_str: str) -> Optional[Any]:
    if not ULTRALYTICS_AVAILABLE or YOLO is None:
        return None
    try:
        return YOLO(model_path_str)
    except Exception as exc:
        st.error(f"Error loading model: {type(exc).__name__}: {exc}")
        return None

def draw_bounding_boxes(image_rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    canvas = image_rgb.copy()
    palette = [
        (27, 67, 50),
        (45, 106, 79),
        (64, 145, 108),
        (217, 83, 79),
        (240, 173, 78),
        (41, 128, 185),
        (142, 68, 173),
    ]
    for det in detections:
        cid = int(det["Class ID"])
        color = palette[cid % len(palette)]
        x1, y1 = int(round(det["X1"])), int(round(det["Y1"]))
        x2, y2 = int(round(det["X2"])), int(round(det["Y2"]))
        label = f"{det['Class Name']} {det['Confidence']:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        ty1 = max(0, y1 - th - 8)
        cv2.rectangle(canvas, (x1, ty1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas

def run_model_inference(
    model: Any,
    image_rgb: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    results = model.predict(
        source=bgr,
        conf=float(conf_threshold),
        iou=float(iou_threshold),
        imgsz=int(imgsz),
        verbose=False,
    )

    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    detections: list[dict[str, Any]] = []
    if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
        boxes_obj = results[0].boxes
        xyxys = boxes_obj.xyxy.cpu().numpy() if hasattr(boxes_obj.xyxy, "cpu") else np.asarray(boxes_obj.xyxy)
        confs = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj.conf, "cpu") else np.asarray(boxes_obj.conf)
        clss = boxes_obj.cls.cpu().numpy().astype(int) if hasattr(boxes_obj.cls, "cpu") else np.asarray(boxes_obj.cls, dtype=int)
        names_dict = getattr(results[0], "names", getattr(model, "names", {})) or {}

        for idx, (cid, score, xyxy) in enumerate(zip(clss, confs, xyxys), start=1):
            cname = names_dict.get(int(cid), f"class_{cid}")
            detections.append({
                "Detection ID": idx,
                "Class ID": int(cid),
                "Class Name": str(cname),
                "Confidence": round(float(score), 4),
                "X1": round(float(xyxy[0]), 2),
                "Y1": round(float(xyxy[1]), 2),
                "X2": round(float(xyxy[2]), 2),
                "Y2": round(float(xyxy[3]), 2),
            })

    annotated = draw_bounding_boxes(image_rgb, detections)
    return annotated, detections, latency_ms

st.markdown(
    """
    <div class="header-box">
        <h2>Road Damage Detection and Localization System</h2>
        <p>Automated Pavement Defect Inspection Interface powered by Ultralytics YOLO</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Inference Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou_thresh = st.slider("IoU Threshold (NMS)", 0.10, 0.90, 0.45, 0.05)
    img_size = st.selectbox("Inference Resolution (px)", [320, 480, 640, 800, 1024], index=2)

    st.markdown("---")
    st.subheader("Model Status")
    resolved_file = resolve_model_path()
    if resolved_file is None:
        st.error("No checkpoint found. Place 'best_road_damage_model.pt' in the working directory.")
        model = None
    else:
        size_mb = resolved_file.stat().st_size / (1024 * 1024)
        model = load_yolo_model(str(resolved_file))
        if model is not None:
            st.success(f"Model Ready: {resolved_file.name} ({size_mb:.2f} MB)")
            names = getattr(model, "names", {})
            st.write(f"Classes ({len(names)}):")
            st.json(names)
        else:
            st.error("Failed to load model weights.")

tab_image, tab_video, tab_camera = st.tabs(["Image Inspection", "Video Processing", "Camera Snapshot"])

# TAB 1: IMAGE INSPECTION
with tab_image:
    st.markdown("### Upload Road Image")
    uploaded_image = st.file_uploader(
        "Choose a road inspection image (JPEG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
    )

    if uploaded_image is not None and model is not None:
        pil_img = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(pil_img)

        col_orig, col_pred = st.columns(2)
        with col_orig:
            st.markdown("**Original Road Image**")
            st.image(img_np, use_container_width=True)

        annotated_np, dets, latency = run_model_inference(
            model, img_np, conf_thresh, iou_thresh, img_size
        )

        with col_pred:
            st.markdown("**Annotated Road Damage Output**")
            st.image(annotated_np, use_container_width=True)

        fps = round(1000.0 / latency, 2) if latency > 0 else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Detections", len(dets))
        m2.metric("Inference Latency", f"{latency:.2f} ms")
        m3.metric("Throughput", f"{fps:.2f} FPS")

        if dets:
            det_df = pd.DataFrame(dets)
            st.markdown("#### Summary by Defect Category")
            sum_rows = []
            for cname, grp in det_df.groupby("Class Name"):
                sum_rows.append({
                    "Class Name": str(cname),
                    "Count": len(grp),
                    "Mean Confidence": round(float(grp["Confidence"].mean()), 4),
                    "Max Confidence": round(float(grp["Confidence"].max()), 4),
                })
            sum_df = pd.DataFrame(sum_rows).sort_values("Count", ascending=False)
            st.dataframe(sum_df, use_container_width=True)

            st.markdown("#### Detailed Defect Coordinates")
            st.dataframe(det_df, use_container_width=True)
        else:
            st.info("No road damage instances detected above the current confidence threshold.")

# TAB 2: VIDEO PROCESSING
with tab_video:
    st.markdown("### Upload Road Inspection Video")
    uploaded_video = st.file_uploader(
        "Choose an inspection video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
    )

    if uploaded_video is not None and model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as in_tmp:
            in_tmp.write(uploaded_video.read())
            in_video_path = in_tmp.name

        cap = cv2.VideoCapture(in_video_path)
        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if src_fps <= 0 or not np.isfinite(src_fps):
            src_fps = 25.0

        st.write(f"Resolution: {src_width}x{src_height} | {src_fps:.1f} FPS | {total_frames} Frames")

        if st.button("Start Video Analysis"):
            out_tmp_path = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_tmp_path, fourcc, src_fps, (src_width, src_height))

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            processed_count = 0
            t_vid_start = time.perf_counter()

            while True:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ann_frame, frame_dets, _ = run_model_inference(
                    model, frame_rgb, conf_thresh, iou_thresh, img_size
                )

                ann_bgr = cv2.cvtColor(ann_frame, cv2.COLOR_RGB2BGR)
                writer.write(ann_bgr)
                processed_count += 1

                if total_frames > 0 and processed_count % 5 == 0:
                    prog_val = min(1.0, processed_count / total_frames)
                    progress_bar.progress(prog_val)
                    status_text.text(f"Processed frame {processed_count} of {total_frames}...")

            cap.release()
            writer.release()
            progress_bar.progress(1.0)
            elapsed_sec = time.perf_counter() - t_vid_start
            actual_fps = round(processed_count / elapsed_sec, 2) if elapsed_sec > 0 else 0.0

            st.success(f"Video analysis complete: {processed_count} frames in {elapsed_sec:.2f}s ({actual_fps} processing FPS).")

            with open(out_tmp_path, "rb") as f_out:
                st.download_button(
                    label="Download Annotated Video (MP4)",
                    data=f_out,
                    file_name="annotated_road_inspection.mp4",
                    mime="video/mp4",
                )

# TAB 3: CAMERA SNAPSHOT
with tab_camera:
    st.markdown("### Real-Time Camera Inspection")
    cam_picture = st.camera_input("Capture a pavement snapshot for instant defect analysis")

    if cam_picture is not None and model is not None:
        cam_pil = Image.open(cam_picture).convert("RGB")
        cam_np = np.array(cam_pil)

        annotated_cam, cam_dets, _ = run_model_inference(
            model, cam_np, conf_thresh, iou_thresh, img_size
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Captured Frame**")
            st.image(cam_np, use_container_width=True)
        with c2:
            st.markdown("**Defect Localization**")
            st.image(annotated_cam, use_container_width=True)

        st.metric("Detected Defect Count", len(cam_dets))
        if cam_dets:
            st.dataframe(pd.DataFrame(cam_dets), use_container_width=True)