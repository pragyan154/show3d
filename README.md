# 🚀 show3d: Turn Your Photos Into Stunning 3D Videos

Ever wished your photos could come alive like magic? ✨ **show3d** lets you do just that! Transform ordinary 2D images into **cinematic 3D parallax slideshow videos**, just like Instagram’s 3D photo feature — but on your own Mac.  

Your favorite moments will pop out with depth, creating an immersive visual experience that feels alive.

---

## 🎯 What It Does

- Automatically separates people/subjects from backgrounds using AI.  
- Generates realistic 3D parallax effects for dynamic slideshows.  
- Smooth transitions between scenes for that cinematic vibe.  
- No complicated editing — just give it your photos, and watch the magic happen.

---

## 🛠️ Technologies Used

- **Python 3.11** – main programming language.  
- **PyTorch** – deep learning backend, uses **MPS** on Mac Silicon.  
- **OpenCV** – video creation, resizing, blending.  
- **Pillow** – image processing.  
- **rembg** – AI-powered human/background extraction.  
- **Segment Anything Model (SAM)** – high-quality segmentation of subjects.  
- **NumPy** – numerical operations on images.  

---

## 🔄 Flow / How It Works

1. **Input Images** – Place all images in `input_photos/`.  
2. **Layer Extraction** – AI separates humans/subjects from the background using `rembg` and `SAM`.  
3. **Background Inpainting** – Any gaps left after extraction are filled automatically.  
4. **Aspect Ratio & Cropping** – Images are resized and cropped to match video resolution.  
5. **Sequence Execution** – `sequence.txt` defines which layers (BG / HUMAN) are shown and in what order.  
6. **Blending & Video Creation** – Layers are blended together with parallax motion, generating smooth frames.  
7. **Output** – Final cinematic video saved as `output_sequence_video.mp4`.

---

## 💻 Who It’s For

- Creators who want eye-catching social media content.  
- Photographers looking to add motion to still images.  
- Anyone who loves immersive visual storytelling.

---

## 🛠️ Prerequisites

- **macOS** (optimized for M1/M2/M3)  
- **Python 3.11.x**  
- **uv** (`pip install uv`)  
- Other dependencies installed via the setup commands below  

> ⚠️ For Windows/Linux: Python 3.11 + PyTorch CPU/CUDA versions may need adjustments. Replace the MPS device setup with `torch.device("cuda")` (if GPU available) or `torch.device("cpu")`.

---

## 🛠️ Installation Steps (Mac)

```bash
# 1. Create project directory
mkdir show3d
cd show3d

# 2. Create and activate Python 3.11 virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# 3. Install Core Libraries
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy opencv-python Pillow

# 4. Install AI Libraries
uv pip install rembg segment-anything

# 5. Download SAM checkpoint
mkdir -p pretrained_models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P pretrained_models/
