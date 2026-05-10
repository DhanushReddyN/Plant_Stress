import streamlit as st
import cv2
import numpy as np
from PIL import Image


def analyze_plant(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    total_pixels = frame.shape[0] * frame.shape[1]

    # Adjusted HSV ranges for better real-world lighting (shadows, indoor light)
    lower_green = np.array([30, 25, 25], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)

    lower_yellow = np.array([15, 30, 30], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    lower_brown = np.array([5, 30, 20], dtype=np.uint8)
    upper_brown = np.array([20, 255, 200], dtype=np.uint8)

    # Masks
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Clean masks a bit
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)

    # Calculate raw pixels
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    brown_pixels = cv2.countNonZero(brown_mask)
    total_plant_pixels = green_pixels + yellow_pixels + brown_pixels

    # Leaf coverage = total visible plant-like area relative to whole image
    leaf_coverage = (total_plant_pixels / total_pixels) * 100

    # Percentages relative to the plant itself (not the whole image)
    if total_plant_pixels > 0:
        green_pct = (green_pixels / total_plant_pixels) * 100
        yellow_pct = (yellow_pixels / total_plant_pixels) * 100
        brown_pct = (brown_pixels / total_plant_pixels) * 100
    else:
        green_pct = yellow_pct = brown_pct = 0.0

    # Dryness score: more weight to brown, some to yellow
    dryness_score = (brown_pct * 0.7) + (yellow_pct * 0.3)

    # Stress score: healthy green reduces stress, yellow/brown increase stress
    stress_score = max(0, min(100, (yellow_pct * 1.2) + (brown_pct * 1.8) - (green_pct * 0.5) + 30))

    # Health classification
    if leaf_coverage < 0.5: # Less than 0.5% of the image is plant
        health = "No Plant Detected"
        stress = "N/A"
        color = (128, 128, 128)
    elif green_pct >= 60 and brown_pct < 10:
        health = "Healthy"
        stress = "Low Stress"
        color = (0, 255, 0)
    elif green_pct >= 30 and brown_pct < 25:
        health = "Moderate"
        stress = "Medium Stress"
        color = (0, 255, 255)
    else:
        health = "Unhealthy"
        stress = "High Stress"
        color = (255, 0, 0) # red in RGB

    return {
        "green_mask": green_mask,
        "yellow_mask": yellow_mask,
        "brown_mask": brown_mask,
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "leaf_coverage": leaf_coverage,
        "dryness_score": dryness_score,
        "stress_score": stress_score,
        "health": health,
        "stress": stress,
        "color": color,
    }

def create_thermal_map(mask):
    blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    thermal = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    return cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB) # Convert to RGB for Streamlit

st.set_page_config(page_title="Plant Stress Analyzer", page_icon="🌿", layout="wide")

st.title("🌿 Plant Stress Analyzer")
st.markdown("Upload an image of a plant or take a picture using your camera to analyze its health and stress levels.")

col1, col2 = st.columns(2)

with col1:
    source_option = st.radio("Choose Image Source:", ["Upload Image", "Take Picture"])
    image_file = None

    if source_option == "Upload Image":
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    else:
        image_file = st.camera_input("Take a picture")

if image_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(image_file)
    
    # Resize image to prevent Streamlit Cloud memory crashes on high-res mobile photos
    max_size = (800, 800)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    frame = np.array(image)
    # Convert RGB to BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing plant..."):
        result = analyze_plant(frame_bgr)
        thermal_map_rgb = create_thermal_map(result["green_mask"])
        
        # Create an overlay
        alpha = 0.45
        thermal_bgr = cv2.cvtColor(thermal_map_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(frame_bgr, 1 - alpha, thermal_bgr, alpha, 0)
        overlay_bgr[result["green_mask"] == 0] = frame_bgr[result["green_mask"] == 0]
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Analysis Results")
    
    # Metrics Dashboard
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Health Status", result["health"])
    m2.metric("Stress Level", result["stress"])
    m3.metric("Dryness Score", f"{result['dryness_score']:.2f}")
    m4.metric("Stress Score", f"{result['stress_score']:.2f}")

    # Color Percentages
    st.markdown("### Color Segmentation")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Green (Healthy)", f"{result['green_pct']:.2f}%")
    p2.metric("Yellow (Warning)", f"{result['yellow_pct']:.2f}%")
    p3.metric("Brown (Dry/Dead)", f"{result['brown_pct']:.2f}%")
    p4.metric("Total Leaf Coverage", f"{result['leaf_coverage']:.2f}%")

    st.divider()

    # Images
    st.markdown("### Visualizations")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(image, caption="Original Image", use_container_width=True)
        st.image(thermal_map_rgb, caption="Thermal Map (Green areas)", use_container_width=True)
    with img_col2:
        st.image(overlay_rgb, caption="Analyzed Plant Overlay", use_container_width=True)
        
        # Displaying masks
        st.markdown("#### Masks")
        mask_col1, mask_col2, mask_col3 = st.columns(3)
        with mask_col1:
            st.image(result["green_mask"], caption="Green Mask", use_container_width=True)
        with mask_col2:
            st.image(result["yellow_mask"], caption="Yellow Mask", use_container_width=True)
        with mask_col3:
            st.image(result["brown_mask"], caption="Brown Mask", use_container_width=True)
