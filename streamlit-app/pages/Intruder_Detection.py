import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import base64
from utils import show_page_info

st.set_page_config(page_title="Intruder Detection", page_icon="🚨")

st.title("Intruder Detection Application")
show_page_info("Intruder_Detection")

def get_video_download_link(video_path, filename, text):
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{text}</a>'
    return href

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.subheader("Detection Settings")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        history = st.slider("History", 10, 500, 200)
        dist_threshold = st.slider("Distance Threshold", 10.0, 2000.0, 400.0, 10.0)
        detect_shadows = st.checkbox("Detect Shadows", value=True)
    with col_s2:
        erosion_kernel_size = st.slider("Erosion Kernel Size", 1, 15, 5, 2)
        motion_threshold = st.slider("Motion Area Threshold (pixels)", 100, 10000, 500, 100)

    if st.button("Process Video"):
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create Background Subtractor
        bg_sub = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist_threshold, detectShadows=detect_shadows)
        
        # Prepare output video
        output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_tfile.name
        
        output_quad_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_quad_path = output_quad_tfile.name

        # Using 'mp4v' for initial writing, will re-encode for web compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Quad view is 2x width and 2x height
        out_quad = cv2.VideoWriter(output_quad_path, fourcc, fps, (width * 2, height * 2))

        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_image = st.empty()

        ksize = (erosion_kernel_size, erosion_kernel_size)
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        intruder_detected = False
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_erode_display = frame.copy()
            
            # Apply background subtraction
            fg_mask = bg_sub.apply(frame)
            
            # Erode to remove noise
            fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
            
            # Find motion area
            motion_area = cv2.findNonZero(fg_mask_erode)
            
            if motion_area is not None:
                x, y, w, h = cv2.boundingRect(motion_area)
                if w * h > motion_threshold:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), red, 2)
                    cv2.putText(frame, "INTRUDER DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
                    
                    cv2.rectangle(frame_erode_display, (x, y), (x + w, y + h), red, 2)
                    intruder_detected = True

            out.write(frame)
            
            # Build Quad View
            # Convert masks to BGR
            fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            fg_mask_erode_bgr = cv2.cvtColor(fg_mask_erode, cv2.COLOR_GRAY2BGR)
            
            # Annotate masks
            cv2.putText(fg_mask_bgr, "Foreground Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(fg_mask_erode_bgr, "Foreground Mask Eroded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_erode_display, "Annotated Eroded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Assemble quad
            top_row = np.hstack((fg_mask_bgr, frame))
            bottom_row = np.hstack((fg_mask_erode_bgr, frame_erode_display))
            quad_frame = np.vstack((top_row, bottom_row))
            
            # Draw yellow lines to separate quad
            cv2.line(quad_frame, (0, height), (width * 2, height), yellow, 2)
            cv2.line(quad_frame, (width, 0), (width, height * 2), yellow, 2)
            
            out_quad.write(quad_frame)
            
            # Show preview every 10 frames to save time/resources
            if frame_idx % 10 == 0:
                preview_image.image(frame, channels="BGR", use_container_width=True)
            
            frame_idx += 1
            progress_bar.progress(frame_idx / total_frames)
            status_text.text(f"Processing frame {frame_idx}/{total_frames}")

        cap.release()
        out.release()
        out_quad.release()
        
        status_text.text("Re-encoding video for web compatibility...")
        
        # Re-encode for web playback using ffmpeg
        web_output_path = output_path.replace('.mp4', '_web.mp4')
        web_quad_path = output_quad_path.replace('.mp4', '_web.mp4')
        
        os.system(f"ffmpeg -y -i {output_path} -vcodec libx264 {web_output_path}")
        os.system(f"ffmpeg -y -i {output_quad_path} -vcodec libx264 {web_quad_path}")
        
        status_text.text("Processing complete!")
        
        if intruder_detected:
            st.error("⚠️ ALERT: Intruder was detected in the video!")
        else:
            st.success("No intruder detected.")

        # Display processed videos
        st.subheader("Processed Video Preview")
        st.video(web_output_path)
        
        st.subheader("Quad View Preview")
        st.video(web_quad_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(get_video_download_link(web_output_path, "processed_intruder_detection.mp4", "Download Processed Video"), unsafe_allow_html=True)
        with col2:
            st.markdown(get_video_download_link(web_quad_path, "intruder_detection_quad.mp4", "Download Quad View Video"), unsafe_allow_html=True)

        # Cleanup
        # os.remove(video_path) # We might want to keep it if we don't re-run
        # We should probably clean up old temp files eventually
    
else:
    st.info("Please upload a video file to start intruder detection.")
