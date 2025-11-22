import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
import torch

# --- CONFIGURATION ---
INPUT_DIR = 'input_photos'
SEQUENCE_FILE = 'sequence.txt'
OUTPUT_VIDEO_PATH = 'output_sequence_video.mp4'
FPS = 30
HOLD_DURATION_SECONDS = 0.5 
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# PyTorch device for Mac Silicon (MPS)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# Initialize Rembg Session (run once for efficiency)
REMBG_SESSION = new_session() 

# --- CORE UTILITY FUNCTIONS (Cropping, Blending) ---

def crop_and_resize_to_fit(image_np, target_width, target_height):
    """
    Scales the image to fill the frame while maintaining aspect ratio, then crops the excess.
    """
    h, w = image_np.shape[:2]
    target_ratio = target_width / target_height
    image_ratio = w / h

    if image_ratio > target_ratio:
        scale_factor = target_height / h
        new_w = int(w * scale_factor)
        new_h = target_height
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        crop_w = new_w - target_width
        start_w = crop_w // 2
        cropped = resized[:, start_w:start_w + target_width]
    else:
        scale_factor = target_width / w
        new_w = target_width
        new_h = int(h * scale_factor)
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        crop_h = new_h - target_height
        start_h = crop_h // 2
        cropped = resized[start_h:start_h + target_height, :]

    return cropped

def blend_layers(base_rgb, top_rgba):
    """Blends a transparent layer (RGBA) onto a base layer (RGB) instantly."""
    blended_rgb = base_rgb.copy()
    
    top_bgr = top_rgba[:, :, :3]
    top_alpha = top_rgba[:, :, 3] / 255.0  # Normalized alpha
    
    # Blending formula: (Foreground * Alpha) + (Background * (1 - Alpha))
    for c in range(3):
        blended_rgb[:, :, c] = (
            top_alpha * top_bgr[:, :, c] +
            (1 - top_alpha) * blended_rgb[:, :, c]
        )
    return blended_rgb

def generate_hold_frames(frame_rgb, total_frames):
    """Repeats a single frame for a specific duration."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return [frame_bgr] * total_frames


# --- PHASE 1: PRE-PROCESSING (Extracting All Layers) ---

def pre_process_all_layers(image_files):
    """Processes all unique input images and stores their BG and Human layers."""
    layer_cache = {}
    total_images = len(image_files)

    for i, file_name in enumerate(image_files):
        print(f"  [1/3] Processing Image {i+1}/{total_images}: {file_name}...")
        image_path = os.path.join(INPUT_DIR, file_name)
        
        img_pil = Image.open(image_path).convert("RGB")
        original_np = np.array(img_pil)
        
        # 1. Extraction (Rembg)
        fg_result_pil = remove(img_pil, session=REMBG_SESSION)
        fg_result_np = np.array(fg_result_pil) # RGBA array

        # 2. Inpainting Mask
        alpha_channel = fg_result_np[:, :, 3]
        mask = (alpha_channel > 0).astype(np.uint8) * 255
        
        # 3. Background Generation (Inpainting)
        bgr_original = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        clean_background_bgr = cv2.inpaint(bgr_original, mask, 3, cv2.INPAINT_TELEA)
        clean_background_rgb = cv2.cvtColor(clean_background_bgr, cv2.COLOR_BGR2RGB)
        
        # 4. Aspect Ratio Correction & Cropping
        human_cutout_cropped = crop_and_resize_to_fit(fg_result_np, VIDEO_WIDTH, VIDEO_HEIGHT)
        clean_background_cropped = crop_and_resize_to_fit(clean_background_rgb, VIDEO_WIDTH, VIDEO_HEIGHT)
        
        # Cache the results using the file name as the key
        layer_cache[file_name] = {
            'BG': np.dstack((clean_background_cropped, np.ones((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8) * 255)),
            'HUMAN': human_cutout_cropped
        }
    
    return layer_cache


# --- PHASE 2: SEQUENCE EXECUTION ---

def execute_sequence(layer_cache, sequence_lines, out, total_frames_hold):
    """Reads sequence file, retrieves layers, blends them, and writes frames."""
    global accumulated_scene_rgb # Use global or pass as argument if preferred
    
    # Initialize the 'Accumulated Scene' (RGB format)
    accumulated_scene_rgb = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    
    for i, line in enumerate(sequence_lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue # Skip comments and empty lines
            
        try:
            # Parse the line: e.g., "BG:img01.jpg"
            layer_type, file_name = line.split(':')
            layer_type = layer_type.strip().upper()
            file_name = file_name.strip()
            
            if file_name not in layer_cache:
                print(f"⚠️ Warning: File '{file_name}' not found in input_photos. Skipping step.")
                continue
                
            if layer_type not in ['BG', 'HUMAN']:
                print(f"⚠️ Warning: Invalid layer type '{layer_type}'. Use 'BG' or 'HUMAN'. Skipping step.")
                continue
                
            # Retrieve the correct layer data (RGBA format)
            top_layer_rgba = layer_cache[file_name][layer_type]
            
            print(f"  [2/3] -> Step {i+1}: INSTANTLY applying {layer_type} from {file_name}...")
            
            # Blend the new layer onto the current accumulated scene
            accumulated_scene_rgb = blend_layers(
                base_rgb=accumulated_scene_rgb, 
                top_rgba=top_layer_rgba
            )
            
            # Hold the resulting scene
            hold_frames = generate_hold_frames(accumulated_scene_rgb, total_frames_hold)
            for frame in hold_frames:
                out.write(frame)
                
        except ValueError:
            print(f"⚠️ Warning: Invalid format in sequence.txt line: '{line}'. Skipping step.")
            continue
            
    print("\n[2/3] Sequence execution complete.")


if __name__ == '__main__':
    if not os.path.exists(SEQUENCE_FILE):
        print(f"🛑 Error: Sequence file '{SEQUENCE_FILE}' not found. Please create it and define the layer order.")
    elif not os.path.exists(INPUT_DIR):
        print(f"🛑 Error: Input directory '{INPUT_DIR}' not found.")
    else:
        # 1. Get list of files and sequence lines
        sequence_lines = []
        with open(SEQUENCE_FILE, 'r') as f:
            sequence_lines = f.readlines()
            
        # Get unique file names mentioned in the sequence file
        unique_files_to_process = set()
        for line in sequence_lines:
            try:
                if not line.strip().startswith('#'):
                    _, file_name = line.split(':')
                    unique_files_to_process.add(file_name.strip())
            except ValueError:
                pass # Ignore badly formatted lines for pre-processing list

        if not unique_files_to_process:
             print("🛑 Error: Sequence file is empty or contains no valid entries.")
        else:
            # 2. PHASE 1: PRE-PROCESS ALL LAYERS
            print("--- PHASE 1: PRE-PROCESSING LAYERS ---")
            layer_cache = pre_process_all_layers(unique_files_to_process)
            
            # 3. Initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            # Calculate hold frames (cast to int to avoid TypeError)
            total_frames_hold = int(FPS * HOLD_DURATION_SECONDS)
            
            # 4. PHASE 2: EXECUTE SEQUENCE
            print("\n--- PHASE 2: EXECUTING SEQUENCE ---")
            execute_sequence(layer_cache, sequence_lines, out, total_frames_hold)
            
            # 5. Finalize Video
            out.release()
            cv2.destroyAllWindows()
            print(f"\n[3/3] ✅ Video successfully created and saved to: {OUTPUT_VIDEO_PATH}")