import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("💰 Advanced Circular Coin Detection (Watershed + OpenCV)")

st.sidebar.header("⚙️ Options")
st.sidebar.info("Upload an image OR click 'Use Example Image' to test.")

# Initialize session state for storing image
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# Load example image
def load_example_image():
    try:
        image = Image.open("3sample.jpg")  # Make sure this file is in same folder
        st.session_state.selected_image = image
    except:
        st.error("❌ Example image '3sample.jpg' not found. Please make sure it's in the same folder.")

# Coin detection using watershed
def detect_coins(image):
    img = np.array(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    colors = np.random.randint(0, 255, (markers.max() + 1, 3))
    circle_count = 0

    for marker in np.unique(markers):
        if marker <= 1:
            continue
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[markers == marker] = 255
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > 0.7:
                circle_count += 1
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(img, center, radius, (0, 255, 0), 2)
                cv2.putText(img, str(circle_count), (center[0] + radius, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    border_size = 50
    border_color = (255, 255, 255)
    img_with_border = cv2.copyMakeBorder(img, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=border_color)
    cv2.putText(img_with_border, f"Total Coins Detected: {circle_count}", (10, border_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img_with_border, circle_count

# Buttons
uploaded_image = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])
use_example = st.button("🧪 Use Example Image")

if use_example:
    load_example_image()

elif uploaded_image is not None:
    st.session_state.selected_image = Image.open(uploaded_image)

# Display image if exists
if st.session_state.selected_image is not None:
    st.image(st.session_state.selected_image, caption="📷 Image to be processed", use_container_width=True)

    if st.button("🔍 Detect Coins"):
        result, count = detect_coins(st.session_state.selected_image)
        st.image(result, caption=f"✅ Detection Result ({count} coin(s) found)", use_container_width=True)
else:
    st.info("👆 Upload an image or click **Use Example Image** to get started.")

# Footer
st.markdown("---")
st.markdown("👨‍💻 *Created by NapsTorres | Powered by OpenCV + Streamlit*")
