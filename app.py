import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import os
import requests
from pathlib import Path
from io import BytesIO
import logging # Import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
TARGET_SIZE = (256, 256) # Input size expected by the model
MODEL_DIR = Path("src") # Directory to store the downloaded model
MODEL_FILENAME = "text_deblur_model.keras"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MODEL_URL = "https://github.com/SREESAIARJUN/text-deblur/releases/download/1.0/text_deblur_model.keras"

# --- Model Loading ---
@st.cache_resource # Cache the loaded model resource
def load_model():
    """Loads the Keras model, downloading it if necessary."""
    model_loaded = False
    if not MODEL_PATH.exists():
        st.info(f"Model not found at {MODEL_PATH}. Downloading from GitHub...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        progress_bar = st.progress(0)
        status_text = st.empty() # Placeholder for download status text

        try:
            with requests.get(MODEL_URL, stream=True, timeout=60) as r: # Increased timeout
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                bytes_downloaded = 0
                chunk_size = 8192

                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            percentage = min(1.0, bytes_downloaded / total_size)
                            progress_bar.progress(percentage)
                            status_text.text(f"Downloading... {bytes_downloaded // 1024}/{total_size // 1024} KB")

            progress_bar.progress(1.0) # Ensure it completes visually
            status_text.text("Download complete.")
            st.success(f"Model downloaded successfully to {MODEL_PATH}.")
            logging.info(f"Model downloaded successfully to {MODEL_PATH}")

        except requests.exceptions.Timeout:
            st.error("Model download timed out. Please check your internet connection and try again.")
            logging.error("Model download timed out.")
            if MODEL_PATH.exists(): os.remove(MODEL_PATH) # Clean up partial download
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            logging.error(f"Error downloading model: {e}")
            if MODEL_PATH.exists(): os.remove(MODEL_PATH) # Clean up partial download
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during model download/saving: {e}")
            logging.error(f"An unexpected error occurred during model download/saving: {e}")
            if MODEL_PATH.exists(): os.remove(MODEL_PATH)
            st.stop()
        finally:
             # Clear the progress bar and status text after download attempt
             progress_bar.empty()
             status_text.empty()

    # --- Try Loading the Model ---
    if MODEL_PATH.exists():
        try:
            logging.info(f"Loading model from {MODEL_PATH}...")
            model = tf.keras.models.load_model(str(MODEL_PATH))
            st.success("Model loaded successfully.")
            logging.info("Model loaded successfully.")
            model_loaded = True
            return model
        except Exception as e:
            st.error(f"Error loading Keras model from {MODEL_PATH}: {e}")
            st.error("The downloaded file might be corrupted or incompatible. Trying to remove it.")
            logging.error(f"Error loading Keras model from {MODEL_PATH}: {e}")
            # Optionally remove the potentially corrupted file
            if MODEL_PATH.exists():
                try:
                    os.remove(MODEL_PATH)
                    st.info(f"Removed potentially corrupted model file: {MODEL_PATH}. Please reload the page to retry download.")
                    logging.info(f"Removed potentially corrupted model file: {MODEL_PATH}")
                except OSError as oe:
                    st.warning(f"Could not remove corrupted model file: {oe}")
                    logging.warning(f"Could not remove corrupted model file: {oe}")
            st.stop()
    else:
        # This case should ideally not be reached if download logic is correct, but added for safety
        st.error("Model file does not exist after download attempt. Cannot proceed.")
        logging.error("Model file does not exist after download attempt.")
        st.stop()


# --- Helper Functions ---
def preprocess_image(img: Image.Image):
    """Converts PIL image to model-compatible numpy array."""
    img = img.convert("L") # Convert to grayscale
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS) # Use LANCZOS for better resize quality
    img_array = img_to_array(img) # Converts to (H, W, C) float32 numpy array
    img_array = img_array / 255.0 # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, H, W, C)
    return img_array

def postprocess_image(pred_array):
    """Converts model output tensor back to a PIL image."""
    # Squeeze batch dimension if present: (1, H, W, C) -> (H, W, C)
    if pred_array.ndim == 4:
        pred_array = pred_array[0]
    # Squeeze channel dimension if it's 1: (H, W, 1) -> (H, W)
    if pred_array.ndim == 3 and pred_array.shape[-1] == 1:
        pred_array = pred_array[:, :, 0]

    # Clip values to [0, 1], scale to [0, 255], and convert to uint8
    pred_img = np.clip(pred_array, 0, 1) * 255.0
    pred_img = pred_img.astype(np.uint8)

    # Create PIL image (mode 'L' for grayscale)
    return Image.fromarray(pred_img, mode="L")

def calculate_psnr(y_true, y_pred):
    """Calculates PSNR between two images (expected range [0, 1])."""
    if y_true.shape != y_pred.shape:
        logging.warning(f"PSNR calculation shape mismatch: {y_true.shape} vs {y_pred.shape}")
        return 0.0 # Cannot compare
    y_true = y_true.astype(np.float64) # Use float64 for precision
    y_pred = y_pred.astype(np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100.0 # Or float('inf') - Identical images
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(y_true, y_pred):
    """Calculates SSIM between two images (expected range [0, 1])."""
    if y_true.shape != y_pred.shape:
        logging.warning(f"SSIM calculation shape mismatch: {y_true.shape} vs {y_pred.shape}")
        return None # Cannot compare
    try:
        # Ensure scikit-image is installed: pip install scikit-image
        from skimage.metrics import structural_similarity as ssim
        # Ensure arrays are float32/64 and in range [0, 1]
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)
        # data_range is the difference between max and min possible pixel values
        # channel_axis=None because we expect grayscale (H, W) input after preprocessing
        # If model outputs (H, W, 1), ssim might need channel_axis=-1, but postprocess handles this.
        return ssim(y_true, y_pred, data_range=1.0, channel_axis=None)
    except ImportError:
        st.warning("scikit-image not installed. Cannot calculate SSIM. Install with: pip install scikit-image", icon="⚠️")
        logging.warning("scikit-image not installed. Cannot calculate SSIM.")
        return None
    except Exception as e:
        st.error(f"Error calculating SSIM: {e}")
        logging.error(f"Error calculating SSIM: {e}")
        return None

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Text Deblurring") # Use wider layout and set title

st.title("📄 Text Image Deblurring")
st.markdown("This app uses a deep learning model (Attention U-Net based) to attempt to remove blur from text images.")

# --- Load Model ---
# This will run only once due to @st.cache_resource
try:
    model = load_model()
except Exception as e: # Catch potential stop errors during loading
    st.error(f"Failed to load the model. The application cannot proceed. Error: {e}")
    logging.error(f"Failed to load the model. The application cannot proceed. Error: {e}")
    st.stop() # Stop execution if model isn't loaded

# --- Display Sample Images Section ---
st.markdown("---")
st.subheader("Examples")
st.write("Sample blurred images and their corresponding sharp (ground truth) versions:")

# --- Get the directory where the script is located ---
# This makes the path finding independent of the current working directory
try:
    # The standard way to get the script's directory
    script_dir = Path(__file__).parent.resolve()
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environment, though less likely with streamlit run)
    script_dir = Path.cwd()
    st.warning(f"Could not determine script directory reliably, using current working directory: {script_dir}. Ensure you run streamlit from the script's directory.", icon="⚠️")

# --- Add Debugging Info (Optional - uncomment to see) ---
# st.write(f"Script directory detected as: `{script_dir}`")
# try:
#     st.write(f"Files found in script directory: `{os.listdir(script_dir)}`")
# except Exception as e:
#     st.write(f"Could not list files in script directory: {e}")
# --- End Debugging Info ---


# Sample filenames expected in the script's directory
sample_pairs = [
    ("blurred_0_000009.png", "sharp_0_000009.png"),
    ("blurred_0_000004.png", "sharp_0_000004.png"),
]

# --- Check if files exist relative to the script directory ---
sample_files_exist = any((script_dir / f).exists() for pair in sample_pairs for f in pair)

if sample_files_exist and len(sample_pairs) > 0 :
    cols = st.columns(len(sample_pairs) * 2) # Two columns per pair
    col_index = 0
    for blurred_fname, sharp_fname in sample_pairs:
        # --- Construct full paths relative to the script directory ---
        blurred_path = script_dir / blurred_fname
        sharp_path = script_dir / sharp_fname
        # --- End Path Construction Change ---

        # --- Debugging paths being checked (Optional - uncomment) ---
        # st.write(f"Checking for blurred: {blurred_path}")
        # st.write(f"Checking for sharp: {sharp_path}")
        # --- End Debugging ---

        # Display Blurred Sample
        with cols[col_index]:
            st.markdown(f"**Blurred**")
            # Use the Path object's exists() method
            if blurred_path.exists():
                try:
                    # Open using the full path
                    img_blurred = Image.open(blurred_path)
                    st.image(img_blurred, caption=f"{blurred_fname}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Cannot load {blurred_fname}: {e}", icon="⚠️")
                    logging.warning(f"Cannot load sample {blurred_fname} from {blurred_path}: {e}")
            else:
                st.caption(f"{blurred_fname}\n(not found at\n{blurred_path})") # Show path if not found
        col_index += 1

        # Display Sharp Sample
        with cols[col_index]:
            st.markdown(f"**Sharp (GT)**")
            # Use the Path object's exists() method
            if sharp_path.exists():
                try:
                     # Open using the full path
                    img_sharp = Image.open(sharp_path)
                    st.image(img_sharp, caption=f"{sharp_fname}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Cannot load {sharp_fname}: {e}", icon="⚠️")
                    logging.warning(f"Cannot load sample {sharp_fname} from {sharp_path}: {e}")
            else:
                st.caption(f"{sharp_fname}\n(not found at\n{sharp_path})") # Show path if not found
        col_index += 1
elif len(sample_pairs) > 0:
    st.warning(f"Could not find any of the sample image files in the script's directory: {script_dir}", icon="⚠️")
    logging.warning(f"Sample image files not found in script directory: {script_dir}")

# --- Main Application Logic ---
st.markdown("---")
st.subheader("Try it yourself!")
st.write("Upload a *blurred* text image (PNG/JPG/JPEG). The model will attempt to restore it.")

uploaded_file = st.file_uploader(
    "Choose a blurred image...",
    type=["png", "jpg", "jpeg"],
    help="Upload your blurred text image here."
    )

if uploaded_file and model: # Check if a file was uploaded and model is loaded
    col_up1, col_up2 = st.columns(2)
    pil_blurred = None # Initialize to None

    with col_up1:
        st.markdown("#### Uploaded Blurred Image")
        try:
            pil_blurred = Image.open(uploaded_file)
            # Display original uploaded image before resizing for processing
            st.image(pil_blurred, caption=f"Original ({uploaded_file.name})", use_container_width=True)
            logging.info(f"User uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}")
            logging.error(f"Error opening uploaded image {uploaded_file.name}: {e}")
            pil_blurred = None # Ensure variable is None if loading fails

    if pil_blurred: # Proceed only if image loaded successfully
        # Process and Predict
        try:
            with st.spinner("Deblurring image..."):
                input_arr = preprocess_image(pil_blurred)
                logging.info(f"Preprocessed image shape for model: {input_arr.shape}")
                pred_arr = model.predict(input_arr)
                logging.info(f"Model prediction output shape: {pred_arr.shape}")
                pil_deblurred = postprocess_image(pred_arr)

            with col_up2:
                st.markdown("#### Deblurred Output")
                st.image(pil_deblurred, caption="Deblurred Result", use_container_width=True)

                # Offer download for the deblurred image
                buf = BytesIO()
                pil_deblurred.save(buf, format="PNG")
                byte_im = buf.getvalue()

                # Create a downloadable filename
                base_name, ext = os.path.splitext(uploaded_file.name)
                download_filename = f"deblurred_{base_name}.png"

                st.download_button(
                    label="Download Deblurred Image (PNG)",
                    data=byte_im,
                    file_name=download_filename,
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"An error occurred during the deblurring process: {e}")
            logging.error(f"An error occurred during the deblurring process: {e}")
            # Optionally display more details in logs or to user if safe

        # --- Optional Ground Truth Comparison ---
        st.markdown("---")
        with st.expander("Optional: Evaluate with Ground Truth"):
            st.write("If you have the corresponding sharp (ground truth) image, upload it here to calculate PSNR and SSIM metrics.")
            gt_file = st.file_uploader(
                "Upload Sharp Ground Truth Image",
                type=["png", "jpg", "jpeg"],
                key="gt_eval", # Unique key for this uploader
                help="Upload the sharp version of the image you uploaded above."
                )

            if gt_file and pil_deblurred: # Need the deblurred result to compare
                try:
                    pil_gt = Image.open(gt_file).convert("L").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    gt_array = np.asarray(pil_gt, dtype=np.float64) / 255.0 # Use float64 for metrics

                    # Use the postprocessed PIL image, converted back to array, resized consistently
                    # Re-resize deblurred image just to be sure it matches GT size for metrics
                    pil_deblurred_resized = pil_deblurred.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    pred_array_eval = np.asarray(pil_deblurred_resized, dtype=np.float64) / 255.0

                    # Ensure arrays have the same shape before metric calculation
                    if gt_array.shape == pred_array_eval.shape:
                        logging.info(f"Calculating metrics between GT ({gt_array.shape}) and Pred ({pred_array_eval.shape})")
                        psnr = calculate_psnr(gt_array, pred_array_eval)
                        ssim = calculate_ssim(gt_array, pred_array_eval)

                        col_met1, col_met2 = st.columns(2)
                        col_met1.metric(label="PSNR (Higher is Better)", value=f"{psnr:.2f} dB")
                        if ssim is not None:
                           col_met2.metric(label="SSIM (Closer to 1 is Better)", value=f"{ssim:.4f}")
                        else:
                           col_met2.write("SSIM calculation failed or is unavailable.")

                        # Display GT side-by-side with result for visual comparison
                        st.markdown("##### Ground Truth vs. Deblurred (Resized for Comparison)")
                        col_comp1, col_comp2 = st.columns(2)
                        col_comp1.image(pil_gt, caption="Ground Truth (Resized)", use_container_width=True)
                        col_comp2.image(pil_deblurred_resized, caption="Deblurred Result (Resized)", use_container_width=True)

                    else:
                        st.error(f"Shape mismatch between Ground Truth ({gt_array.shape}) and Prediction ({pred_array_eval.shape}) after resizing. Cannot calculate metrics.", icon="🚨")
                        logging.error(f"Shape mismatch GT ({gt_array.shape}) vs Pred ({pred_array_eval.shape})")

                except Exception as e:
                    st.error(f"Error processing ground truth image or calculating metrics: {e}", icon="🚨")
                    logging.error(f"Error processing ground truth image or calculating metrics: {e}")


# --- Footer ---
st.markdown("""
---
*Model based on Attention U-Net architecture. Powered by [Streamlit](https://streamlit.io/) and [TensorFlow/Keras](https://www.tensorflow.org/)*
""")
