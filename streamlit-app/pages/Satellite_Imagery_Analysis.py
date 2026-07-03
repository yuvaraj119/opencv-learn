import streamlit as st
import cv2
import numpy as np
from osgeo import gdal
import tempfile
import os
import matplotlib.pyplot as plt
from utils import show_page_info

st.set_page_config(page_title="Analyzing Satellite Imagery", page_icon="🛰️")

st.title("Analyzing Satellite Imagery using GeoTIFF")
show_page_info("Satellite_Imagery_Analysis")

def get_NDVI(ds):
    # In LandSat5, Red is Band 3 and NIR is Band 4
    r = ds.GetRasterBand(3).ReadAsArray().astype(float)
    nir = ds.GetRasterBand(4).ReadAsArray().astype(float)
    # Avoid division by zero
    denominator = nir + r
    denominator[denominator == 0] = 1e-10
    ndvi = (nir - r) / denominator
    return ndvi

def normalize255(ndvi):
    ndvi_norm = (ndvi + 1) / 2 * 255
    return ndvi_norm.astype(np.uint8)

def get_NDVI_mask(ndvi_norm, threshold=200):
    _, mask = cv2.threshold(ndvi_norm, threshold, 255, cv2.THRESH_BINARY)
    return mask

def percent_forest(mask):
    c = cv2.countNonZero(mask)
    t = mask.shape[0] * mask.shape[1]
    return round((c / t) * 100, 4)

uploaded_files = st.file_uploader("Upload GeoTIFF (.tif) files (up to 5)", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Please upload up to 5 files only. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]

    st.subheader("Analysis Settings")
    
    # NDVI Range for Normalization/Display
    st.subheader("NDVI Normalization Range")
    ndvi_min_input = st.slider("NDVI Min", -1.0, 1.0, -1.0, 0.05)
    ndvi_max_input = st.slider("NDVI Max", -1.0, 1.0, 1.0, 0.05)
    
    # Color Range (Colormap)
    st.subheader("Visualization")
    cmap_options = ['RdYlGn', 'viridis', 'magma', 'inferno', 'plasma', 'gray']
    selected_cmap = st.selectbox("Colormap", cmap_options, index=0)
    
    # Threshold Range for Mask
    st.subheader("Thresholding")
    # In existing code, threshold was 200 on a 0-255 scale.
    # User asked for "threshold range"
    threshold_min = st.slider("Mask Threshold Min", 0, 255, 200)
    threshold_max = st.slider("Mask Threshold Max", 0, 255, 255)

    st.subheader("Band Visualization")
    band_names = [
        "1. Blue",
        "2. Green",
        "3. Red",
        "4. Near Infrared (NIR)",
        "5. Short-wave Infrared (SWIR) 1",
        "6. Thermal",
        "7. Short-wave Infrared (SWIR) 2"
    ]
    show_bands = st.multiselect("Select Bands to View", band_names)

    apply_btn = st.button("Apply Analysis")
    
    if "sat_applied" not in st.session_state:
        st.session_state.sat_applied = False
    
    # Reset applied state if settings change
    if "prev_ndvi_min" not in st.session_state or st.session_state.prev_ndvi_min != ndvi_min_input or \
       "prev_ndvi_max" not in st.session_state or st.session_state.prev_ndvi_max != ndvi_max_input or \
       "prev_cmap" not in st.session_state or st.session_state.prev_cmap != selected_cmap or \
       "prev_t_min" not in st.session_state or st.session_state.prev_t_min != threshold_min or \
       "prev_t_max" not in st.session_state or st.session_state.prev_t_max != threshold_max or \
       "prev_show_bands" not in st.session_state or st.session_state.prev_show_bands != show_bands:
        st.session_state.sat_applied = False
        st.session_state.prev_ndvi_min = ndvi_min_input
        st.session_state.prev_ndvi_max = ndvi_max_input
        st.session_state.prev_cmap = selected_cmap
        st.session_state.prev_t_min = threshold_min
        st.session_state.prev_t_max = threshold_max
        st.session_state.prev_show_bands = show_bands

    if apply_btn:
        st.session_state.sat_applied = True

    results = []

    if st.session_state.sat_applied:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                ds = gdal.Open(tmp_path)
                if ds is None:
                    st.error(f"Could not open {uploaded_file.name}")
                    continue
                
                num_bands = ds.RasterCount
                if num_bands < 4:
                    st.warning(f"{uploaded_file.name} has fewer than 4 bands. Skipping NDVI.")
                    continue

                ndvi = get_NDVI(ds)
                
                # Normalize NDVI to 0-255 based on user-defined NDVI range
                # Formula: (val - min) / (max - min) * 255
                ndvi_range = ndvi_max_input - ndvi_min_input
                if ndvi_range == 0: ndvi_range = 1e-10
                ndvi_norm = np.clip((ndvi - ndvi_min_input) / ndvi_range * 255, 0, 255).astype(np.uint8)
                
                # Get Mask based on threshold range
                mask = cv2.inRange(ndvi_norm, threshold_min, threshold_max)
                perc = percent_forest(mask)
                
                # Extract requested bands
                extracted_bands = {}
                for band_str in show_bands:
                    band_num = int(band_str.split('.')[0])
                    if band_num <= num_bands:
                        extracted_bands[band_str] = ds.GetRasterBand(band_num).ReadAsArray()
                    else:
                        extracted_bands[band_str] = None # Band not available in this file

                results.append({
                    "name": uploaded_file.name,
                    "ndvi": ndvi,
                    "mask": mask,
                    "percentage": perc,
                    "bands": extracted_bands
                })
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text("Processing complete!")
        progress_bar.empty()

    # Display results
    if results:
        for i, res in enumerate(results):
            st.subheader(f"File {i+1}: {res['name']}")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                im = ax.imshow(res['ndvi'], cmap=selected_cmap, vmin=ndvi_min_input, vmax=ndvi_max_input)
                plt.colorbar(im, ax=ax)
                ax.set_title("NDVI Index")
                st.pyplot(fig)
            
            with col2:
                st.image(res['mask'], caption=f"Vegetation Mask ({res['percentage']}%)", use_container_width=True)
                st.metric("Detected Vegetation", f"{res['percentage']}%")
                if i > 0:
                    change = round(res['percentage'] - results[i-1]['percentage'], 4)
                    st.metric("Change from Previous", f"{change}%", delta=change, delta_color="inverse")

            # Display individual bands if selected
            if res['bands']:
                st.write("### Selected Bands")
                # Group bands in columns
                band_cols = st.columns(2)
                for idx, (b_name, b_data) in enumerate(res['bands'].items()):
                    with band_cols[idx % 2]:
                        if b_data is not None:
                            fig_b, ax_b = plt.subplots()
                            im_b = ax_b.imshow(b_data, cmap=selected_cmap)
                            plt.colorbar(im_b, ax=ax_b)
                            ax_b.set_title(b_name)
                            st.pyplot(fig_b)
                        else:
                            st.warning(f"{b_name} not available in this file.")

        if len(results) > 1:
            st.divider()
            st.header("Comparative Analysis")
            summary_data = {
                "File": [r['name'] for r in results],
                "Vegetation Coverage (%)": [r['percentage'] for r in results]
            }
            st.table(summary_data)
    elif st.session_state.sat_applied == False:
        st.info("Adjust settings and click 'Apply Analysis' to process files.")

else:
    st.info("Please upload GeoTIFF files to begin.")