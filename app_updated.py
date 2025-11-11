import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import time

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Crowd Monitor", layout="wide", page_icon="🤖")

# ================== MODERN DARK UI THEME ==================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0a0a0f !important;
    color: #f2f2f2 !important;
    font-family: 'Poppins', sans-serif;
}

.stApp { background-color: #0a0a0f; }

h1, h2, h3, h4, h5, h6 {
    color: #f2f2f2 !important;
    text-shadow: 0 0 20px rgba(0,255,255,0.2);
    letter-spacing: 0.5px;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #00FFFF, #7B61FF);
    box-shadow: 0 0 10px #00FFFF;
}

.kpi-box {
    background: rgba(25, 25, 35, 0.8);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(0,255,255,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(0,255,255,0.05);
    transition: all 0.3s ease-in-out;
}
.kpi-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 25px rgba(123,97,255,0.3);
}
.kpi-title {
    font-size: 1rem;
    text-transform: uppercase;
    color: #9e9eff;
    letter-spacing: 1px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00FFFF, #7B61FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.upload-box {
    border: 2px dashed rgba(0,255,255,0.3);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: rgba(20,20,30,0.6);
}
.footer {
    text-align: center;
    color: #7B61FF;
    margin-top: 2rem;
    font-size: 0.9rem;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<h1 style='text-align:center;'>🤖 AI Crowd Monitor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#00FFFF;'>Real-time people detection, analytics & monitoring powered by YOLOv8</h4>", unsafe_allow_html=True)
st.write("")

# ================== MODEL LOAD ==================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ================== FILE UPLOAD ==================
st.markdown("<div class='upload-box'>📂 Drag & Drop or Upload a Video File (.mp4, .avi, .mkv)</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = video_path.replace(".mp4", "_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # KPIs
    k1, k2, k3 = st.columns(3)
    total_kpi = k1.empty()
    fps_kpi = k2.empty()
    unique_kpi = k3.empty()

    progress = st.progress(0)
    display_frame = st.empty()

    total_people = set()
    frame_count = 0
    start_time = time.time()

    # ================== MAIN LOOP ==================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)[0]
        people_in_frame = 0

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            people_in_frame += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            total_people.add((cx // 20, cy // 20))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        fps_calc = frame_count / (time.time() - start_time)

        total_kpi.markdown(f"<div class='kpi-box'><div class='kpi-title'>👥 People in Frame</div><div class='kpi-value'>{people_in_frame}</div></div>", unsafe_allow_html=True)
        fps_kpi.markdown(f"<div class='kpi-box'><div class='kpi-title'>⚡ Average FPS</div><div class='kpi-value'>{fps_calc:.2f}</div></div>", unsafe_allow_html=True)
        unique_kpi.markdown(f"<div class='kpi-box'><div class='kpi-title'>🎞️ Total Unique</div><div class='kpi-value'>{len(total_people)}</div></div>", unsafe_allow_html=True)

        display_frame.image(frame, channels="BGR", use_container_width=True)
        out.write(frame)
        progress.progress(frame_count / total_frames)

    cap.release()
    out.release()

    # ================== POST-PROCESS ==================
    st.markdown("---")
    st.success(f"✅ Analysis complete — {len(total_people)} unique people detected.")
    st.markdown("<h4 style='color:#00FFFF;'>⬇️ Download Processed Video</h4>", unsafe_allow_html=True)
    with open(output_path, "rb") as file:
        st.download_button("📥 Download Output", file, file_name="ai_crowd_output.mp4", use_container_width=True)

# ================== FOOTER ==================
st.markdown("<div class='footer'>© 2025 AI Crowd Monitor | Powered by YOLOv8 + Streamlit</div>", unsafe_allow_html=True)