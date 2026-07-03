import streamlit as st
import qrcode
import qrcode.constants
from PIL import Image
import io
from utils import show_page_info

st.set_page_config(page_title="QR Code Generation", page_icon="🔗")

st.title("QR Code Generation")
show_page_info("QR_Code_Generation")

# Input for QR code data
data = st.text_area("Enter the data to encode in the QR code", "https://www.example.com")

# Options for customization
col1, col2, col3 = st.columns(3)
with col1:
    version = st.slider("Version (1-40)", 1, 40, 1, help="Higher version for more data")
with col2:
    box_size = st.slider("Box Size", 1, 20, 10, help="Size of each box in pixels")
with col3:
    border = st.slider("Border", 0, 10, 4, help="Border thickness")

# Error correction level
error_correction = st.selectbox("Error Correction Level", ["L", "M", "Q", "H"], index=1,
                                help="L: 7%, M: 15%, Q: 25%, H: 30%")

# Color options
col4, col5 = st.columns(2)
with col4:
    fill_color = st.color_picker("Fill Color", "#000000")
with col5:
    back_color = st.color_picker("Background Color", "#FFFFFF")

if st.button("Generate QR Code"):
    # Map error correction
    ec_dict = {"L": qrcode.constants.ERROR_CORRECT_L,
               "M": qrcode.constants.ERROR_CORRECT_M,
               "Q": qrcode.constants.ERROR_CORRECT_Q,
               "H": qrcode.constants.ERROR_CORRECT_H}

    # Create QR code
    qr = qrcode.QRCode(
        version=version,
        error_correction=ec_dict[error_correction],
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Create image
    img = qr.make_image(fill_color=fill_color, back_color=back_color)

    # Convert to PIL Image
    pil_img = img.convert("RGB")

    # Display
    st.image(pil_img, caption="Generated QR Code")

    # Download option
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download QR Code",
        data=byte_im,
        file_name="qr_code.png",
        mime="image/png"
    )