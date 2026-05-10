# YOLO Object Detection

YOLO object detection assignment code using TensorFlow/Keras. The script filters YOLO boxes, computes IoU, applies non-max suppression, loads a YOLO model, and writes an annotated output image.

## Setup

Clone the repository:

```bash
git clone https://github.com/francisnatusm/Yolo-Object-Detection.git
cd Yolo-Object-Detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model And Assets

Large model files and generated images are intentionally ignored by git. To run prediction, place the required YOLO model assets under `model_data/`, for example:

```text
model_data/saved_model.pb
model_data/variables/
```

Also place an input image at:

```text
images/test.jpg
```

## Run

```bash
python yolo_detection.py
```

The annotated image is saved to `out/test.jpg`.