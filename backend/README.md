# MTM Industrial Motion Analysis Pipeline — Phase 1
## YOLOv8-Pose → Skeleton Sequence Builder → ST-GCN Ready

```
VIDEO INPUT
    ↓
YOLOv8-Pose  (keypoints extraction)
    ↓
ByteTrack    (multi-person tracking)
    ↓
Skeleton Sequence Builder  (time series → ST-GCN format)
    ↓
[PHASE 2] ST-GCN  (action recognition)
    ↓
[PHASE 3] MS-TCN  (temporal smoothing)
    ↓
[PHASE 4] Claude API  (MTM code formatting)
```

---

## 📁 Project Structure

```
mtm_pipeline/
├── README.md
├── requirements.txt
├── configs/
│   └── pipeline_config.yaml       # All tunable parameters
├── core/
│   ├── pose_extractor.py          # YOLOv8-Pose wrapper
│   ├── tracker.py                 # ByteTrack multi-person tracking
│   ├── skeleton_builder.py        # Skeleton sequence builder (ST-GCN input)
│   ├── action_classifier.py       # Rule-based MTM classifier (pre ST-GCN)
│   └── mtm_formatter.py           # Claude API MTM code formatter
├── models/
│   └── model_manager.py           # Model download + cache manager
├── utils/
│   ├── video_utils.py             # Video I/O, frame extraction
│   ├── visualizer.py              # Skeleton overlay + debug rendering
│   └── exporter.py                # JSON / CSV / TXT output exporter
├── scripts/
│   ├── run_pipeline.py            # Main CLI entry point
│   └── batch_process.py           # Batch process multiple videos
└── outputs/                       # Auto-created output directory
```

---

## ⚙️ Setup

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on a video
```bash
python scripts/run_pipeline.py --video path/to/video.mp4
```

### 4. Run with debug visualization
```bash
python scripts/run_pipeline.py --video path/to/video.mp4 --debug --save-video
```

### 5. Batch process a folder
```bash
python scripts/batch_process.py --folder path/to/videos/ --output outputs/
```

---

## 📤 Output Format

```
TITLE
WALK 11-15 STEPS
GET + HOLD OBJECT
WALK 8-10 STEPS
HOLD + PUT OBJECT
GRASP + PLACE OBJECT
HOLD + SLIDE OBJECT (M3)
...
```

Also exports:
- `output.json`  — full skeleton sequences with timestamps
- `output.csv`   — frame-by-frame keypoint data
- `mtm_codes.txt` — final MTM sequence (as above)

---

## 🔧 Configuration (`configs/pipeline_config.yaml`)
All parameters are tunable without touching code.

---

## 🗺️ Roadmap
- [x] Phase 1: YOLOv8-Pose + Skeleton Builder
- [ ] Phase 2: ST-GCN action classifier (training pipeline)
- [ ] Phase 3: MS-TCN temporal smoothing
- [ ] Phase 4: Claude API MTM formatter
- [ ] Phase 5: Multi-camera support
