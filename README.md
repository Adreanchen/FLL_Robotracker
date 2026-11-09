# FLL Robot Game Livestream & YOLOv8 Vision Tracker

A unified real-time computer vision system for FIRST LEGO League (FLL) tables. Combines an automated dual-table OBS/Livestream video wall, high-speed FastAPI vision backend, mobile match control panel, and custom GPU-accelerated YOLOv8 LEGO robot tracker.

---

## 1. Quick Start (Windows)

### 1.1 One-Time Setup & GPU Requirements

1. **Install CUDA-accelerated PyTorch for your NVIDIA GPU**:
   ```powershell
   py -3.12 -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Install all remaining project dependencies**:
   ```powershell
   py -3.12 -m pip install -r requirements.txt
   ```

3. **Verify GPU availability**:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
   ```

*(Frontend `node_modules` are installed automatically the first time you run `python run.py`.)*

---

### 1.2 Run Everything with One Command

From the repository root folder, run:

```powershell
python run.py
```

This single command starts both servers concurrently:
- **Video Wall / Livestream Overlay (React)**: `http://localhost:3000`
- **Phone / PC Match Control & ROI Calibration UI**: `http://<YOUR-PC-IP>:8002`

Press `Ctrl+C` in your terminal to stop all processes gracefully.

---

## 2. Custom YOLOv8 Robot Training Workflow

Fine-tuning a custom YOLOv8 model (`best.pt`) trains the vision tracker to pinpoint your LEGO robot on the competition mat while ignoring human hands, arms, table reflections, and spectators.

```
┌─────────────────────────────────┐
│ 1. Extract Frames from Video    │ python run.py extract --video TestVideo_2.mp4
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
│ 4. Train Model on GPU (cuda:0)  │ python run.py train
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

```powershell
python 1_extract_frames.py --video TestVideo_2.mp4 --interval 150
```
*(Or `python run.py extract --video TestVideo_2.mp4 --interval 150`)*

- All sampled JPG frames are saved into `dataset/raw/`.
- `--interval 150` extracts every 150th frame (~every 5 seconds at 30 FPS). Adjust the interval as needed (e.g., 60 for every 2 seconds).

---

### Step 2: Annotate Images (Roboflow / CVAT)

1. Upload the frames from `dataset/raw/` to [Roboflow](https://roboflow.com/) or CVAT.
2. Draw bounding boxes around the **`robot`** class only:
   - **Do annotate**: LEGO Spike Prime / Mindstorms robot, including all active attachments.
   - **Do NOT annotate**: Human hands, arms, mission models, or referee clipboards. Leaving them unannotated teaches the neural network to treat them as background.
3. In Roboflow, click **Generate Version** and then **Export Dataset**.
4. Choose the format: **YOLOv8**.

---

### Step 3: Where to Paste Labeled Images & Annotations

When you extract the exported zip from Roboflow, paste the contents into the [`dataset/`](./dataset/) directory following this structure:

```
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
> `<class_id> <x_center> <y_center> <width> <height>` (normalized between `0.0` and `1.0`).
> For this single-class robot tracker, `<class_id>` is always `0`.

---

### Step 4: Train on GPU (`cuda:0`)

Once images and labels are in `dataset/train/` and `dataset/valid/`, start the GPU training:

```powershell
python run.py train
```
*(Or directly: `python 2_train_model.py --epochs 50 --batch 16 --imgsz 640`)*

- Automatically leverages your **NVIDIA GeForce RTX GPU (`cuda:0`)** with mixed-precision (`amp=True`).
- Trains for 50 epochs on single class (`single_cls=True`).
- When training finishes, the best weights are automatically saved to:
  - `best.pt` (root folder)
  - `models/best.pt`

---

### Step 5: Test & Run Live Tracking

Test the trained model immediately in a standalone OpenCV tracking window:

```powershell
python run.py track
```
*(Or `python 3_live_tracking.py --weights best.pt --source 0`)*

Hotkeys in the tracking window:
- `v`: Toggle view modes (Side-by-Side Split vs. Zoomed vs. HUD only)
- `r`: Reset camera zoom & smooth tracking state
- `+` / `-`: Increase / decrease confidence threshold
- `m`: Toggle motion requirement gate
- `q` / `ESC`: Quit tracker
