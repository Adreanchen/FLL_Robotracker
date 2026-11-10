# FLL Robot Game Livestream & YOLOv8 Vision Tracker

A unified real-time computer vision system for FIRST LEGO League (FLL) tables. Combines an automated dual-table OBS/Livestream video wall, high-speed FastAPI vision backend, mobile match control panel, and custom-trained YOLOv8 LEGO robot tracker.

Supports **Windows**, **Linux**, and **macOS** with automatic hardware acceleration (**NVIDIA CUDA**, **Apple Silicon MPS**, and **CPU fallback**).

---

## 1. Installation & Environment Setup

The system works across all platforms and automatically selects the fastest available hardware. Choose your OS and hardware setup below:

### 1.1 Step 1: Install PyTorch for Your Hardware

#### Case A: NVIDIA GPU (CUDA 12.4 Acceleration) — *Windows & Linux*
Recommended for fastest training (RTX 30/40 series, GTX, etc.):

- **Windows**:
  ```powershell
  py -3.12 -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
  ```
- **Linux**:
  ```bash
  python3 -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
  ```

---

#### Case B: Standard CPU (No Dedicated NVIDIA GPU) — *Windows, Linux & Mac*
If you don't have an NVIDIA GPU, standard CPU-optimized wheels run tracking and training without any CUDA drivers:

- **Windows**:
  ```powershell
  py -3.12 -m pip install torch torchvision
  ```
- **Linux / macOS**:
  ```bash
  python3 -m pip install torch torchvision
  ```

---

#### Case C: Apple Silicon Mac (M1 / M2 / M3 / M4) — *macOS Metal (MPS)*
The codebase runs natively on macOS without any code modifications. PyTorch for macOS ARM64 automatically leverages Apple's GPU via Metal Performance Shaders (MPS):

- **macOS**:
  ```bash
  python3 -m pip install torch torchvision
  ```

---

#### Case D: AMD GPU (ROCm) — *Linux*
If you are running an AMD Radeon GPU under Linux with ROCm:

- **Linux**:
  ```bash
  python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
  ```

---

### 1.2 Step 2: Install Project Requirements

Install the remaining dependencies (OpenCV, FastAPI, Ultralytics YOLOv8, etc.):

- **Windows**:
  ```powershell
  py -3.12 -m pip install -r requirements.txt
  ```
- **Linux / macOS**:
  ```bash
  python3 -m pip install -r requirements.txt
  ```

---

### 1.3 Step 3: Verify Hardware Acceleration

Check that PyTorch detects your platform's accelerator:

- **Windows**:
  ```powershell
  python -c "import torch; print('Device:', 'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Apple MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU')"
  ```
- **Linux / macOS**:
  ```bash
  python3 -c "import torch; print('Device:', 'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Apple MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU')"
  ```

*(Frontend `node_modules` are installed automatically the first time you run `python run.py` / `python3 run.py`.)*

---

## 2. Run Everything with One Command

From the root repository folder:

- **Windows**:
  ```powershell
  python run.py
  ```
- **Linux / macOS**:
  ```bash
  python3 run.py
  ```

This single command starts both servers concurrently:
- **Video Wall / Livestream Overlay (React)**: `http://localhost:3000`
- **Phone / PC Match Control & ROI Calibration UI**: `http://<YOUR-LAN-IP>:8002`

Press `Ctrl+C` in your terminal to stop all processes gracefully.

---

## 3. Pre-Trained Weights (`best.pt`)

The repository includes pre-trained weights for the FLL LEGO robot:
- [`best.pt`](./best.pt) (workspace root)
- [`models/best.pt`](./models/best.pt)

If you clone this repository, you **do not need to train a model**. `python run.py track` and `python run.py` detect and load `best.pt` automatically.

---

## 4. Custom YOLOv8 Robot Training Workflow (Optional)

If you want to train a custom model tailored to your specific robot geometry and lighting:

```
┌─────────────────────────────────┐
│ 1. Extract Frames from Video    │ python run.py extract --video match.mp4
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 2. Annotate in Roboflow / CVAT │ Label class 0 ("robot") only
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 3. Paste into dataset/ Folders  │ train/images, train/labels, valid/images, valid/labels
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 4. Train Model (Auto Device)    │ python run.py train
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 5. Auto-Generates best.pt       │ Picked up automatically by live tracker & backend
└─────────────────────────────────┘
```

---

### Step 1: Extract Frames from Match Recording

Extract representative frames from any `.mp4` video (e.g. `TestVideo_2.mp4`):

- **Windows**:
  ```powershell
  python 1_extract_frames.py --video TestVideo_2.mp4 --interval 150
  ```
- **Linux / macOS**:
  ```bash
  python3 1_extract_frames.py --video TestVideo_2.mp4 --interval 150
  ```

Sampled frames are written to `dataset/raw/`. Adjust `--interval` as needed (e.g. `60` for every 2 seconds).

---

### Step 2: Annotate Images (Roboflow / CVAT)

1. Upload the frames from `dataset/raw/` to [Roboflow](https://roboflow.com/) or CVAT.
2. Draw bounding boxes around the **`robot`** class only:
   - **Do annotate**: LEGO Spike Prime / Mindstorms robot, including all active attachments.
   - **Do NOT annotate**: Human hands, arms, mission models, or referee clipboards. Leaving them unannotated teaches the neural network to treat them as background.
3. In Roboflow, click **Generate Version** and export in **YOLOv8** format.

---

### Step 3: Where to Paste Labeled Images & Annotations

Extract the Roboflow zip file directly into the [`dataset/`](./dataset/) directory following this structure:

```text
dataset/
├── data.yaml                     # Dataset config (nc: 1, names: [robot])
├── train/
│   ├── images/                   # Training JPG/PNG images
│   │   ├── fll_frame_000150.jpg
│   │   └── ...
│   └── labels/                   # Matching YOLO TXT annotations
│       ├── fll_frame_000150.txt
│       └── ...
└── valid/
    ├── images/                   # Validation JPG/PNG images
    │   ├── fll_frame_001200.jpg
    │   └── ...
    └── labels/                   # Matching YOLO TXT annotations
        ├── fll_frame_001200.txt
        └── ...
```

> [!NOTE]
> Each `.txt` label file contains lines formatted as:
> `0 <x_center> <y_center> <width> <height>` (normalized between `0.0` and `1.0`).
> For this single-class robot tracker, the class index is always `0`.

---

### Step 4: Train Model (Auto Device: CUDA / MPS / CPU)

Once images and labels are in `dataset/train/` and `dataset/valid/`:

- **Windows**:
  ```powershell
  python run.py train
  ```
  *(Or: `python 2_train_model.py --epochs 50 --batch 16 --imgsz 640`)*
- **Linux / macOS**:
  ```bash
  python3 run.py train
  ```
  *(Or: `python3 2_train_model.py --epochs 50 --batch 16 --imgsz 640`)*

The training script automatically chooses the best available hardware:
1. **NVIDIA CUDA GPU (`cuda:0`)** with FP16 mixed precision.
2. **Apple Silicon GPU (`mps`)** on macOS M1/M2/M3/M4.
3. **CPU** fallback on any standard workstation.

When training completes, the best weights are automatically saved to [`best.pt`](./best.pt) and [`models/best.pt`](./models/best.pt).

---

### Step 5: Test & Run Live Tracking

Test tracking with a webcam or video file:

- **Windows**:
  ```powershell
  python run.py track
  ```
- **Linux / macOS**:
  ```bash
  python3 run.py track
  ```

#### Live Tracking Hotkeys:
| Key | Action |
| :--- | :--- |
| `v` | Cycle view modes (Split HUD+Zoom / Zoom Only / HUD Only) |
| `r` | Reset camera zoom & smooth tracking state |
| `+` / `-` | Increase / decrease confidence threshold |
| `m` | Toggle movement requirement gate |
| `q` / `ESC` | Quit tracker |
