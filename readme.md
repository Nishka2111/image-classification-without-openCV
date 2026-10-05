# Fashion-MNIST Object Detection Without OpenCV

A custom deep learning project that performs **multi-object detection on synthetic 224×224 images using Fashion-MNIST**.

Instead of classifying a single Fashion-MNIST image, this project creates synthetic images containing **2–4 randomly placed fashion objects** and trains a custom CNN to simultaneously:

- Detect whether an object exists
- Predict its bounding box
- Classify the object into one of 10 Fashion-MNIST classes

The entire detection pipeline is implemented using **PyTorch, NumPy, PIL, and Matplotlib**, without using OpenCV or a pre-built object detection framework.

---

## Project Overview

Fashion-MNIST normally contains individual **28×28 grayscale images** belonging to 10 clothing categories.

This project transforms the classification dataset into an object detection problem.

Individual Fashion-MNIST images are randomly selected, resized slightly, and placed onto a **224×224 grayscale canvas**. Multiple objects are placed on the same canvas while limiting excessive overlap.

For example, a generated image may contain:

- A sneaker
- A shirt
- A bag

Each object has its own:

- Class label
- Bounding box

The model then predicts all of these objects using a **7×7 detection grid**.

---

## Classes

The model detects the following 10 Fashion-MNIST classes:

| ID | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## How the Dataset Is Created

The original Fashion-MNIST images are only 28×28 pixels, so the project generates a synthetic detection dataset.

For every synthetic sample:

1. A blank **224×224 grayscale canvas** is created.
2. Between **2 and 4 Fashion-MNIST images** are randomly selected.
3. Each object is randomly scaled between approximately **0.85× and 1.20×**.
4. Each object is randomly positioned on the canvas.
5. Objects with excessive overlap are rejected.
6. The selected objects are placed onto the canvas.
7. Their bounding boxes and class labels are stored as ground truth.

This means the model is trained on a new synthetic arrangement of objects whenever a dataset sample is requested.

---

## Detection Grid

The image is divided conceptually into a **7×7 grid**.

Each grid cell predicts:

```text
Objectness
Bounding Box
Class
```

For every cell, the model outputs:

```text
5 + 10 = 15 values
```

### The 5 detection values

```text
1 → Objectness
4 → Bounding box
```

The bounding box contains:

```text
x_offset
y_offset
width
height
```

The remaining 10 values represent the class logits for the 10 Fashion-MNIST classes.

Therefore, the model output has the structure:

```text
7 × 7 × 15
```

---

## Target Encoding

The ground-truth bounding boxes are converted into the same 7×7 grid representation used by the CNN.

For every object:

1. The center of its bounding box is calculated.
2. The grid cell containing that center is identified.
3. The center's position inside that cell is converted into `x_offset` and `y_offset`.
4. Width and height are normalized relative to the 224×224 image.
5. The object's class is stored in that grid cell.

The project creates three target arrays:

```text
objectness
box_targets
class_targets
```

These are used as the ground truth during training.

---

## Model Architecture

The detector uses a custom convolutional neural network implemented in PyTorch.

### Backbone

The CNN contains **5 convolutional layers**:

```text
Input
  ↓
Conv2D 1 → 32
  ↓
BatchNorm
  ↓
ReLU
  ↓
Conv2D 32 → 64
  ↓
BatchNorm
  ↓
ReLU
  ↓
Conv2D 64 → 128
  ↓
BatchNorm
  ↓
ReLU
  ↓
Conv2D 128 → 256
  ↓
BatchNorm
  ↓
ReLU
  ↓
Conv2D 256 → 256
  ↓
BatchNorm
  ↓
ReLU
```

All convolutional layers use:

```text
Kernel size = 3×3
Stride = 2
Padding = 1
```

The stride of 2 progressively reduces the spatial resolution while increasing the number of feature channels.

### Detection Head

A final **1×1 convolution** converts the extracted features into the required detection predictions.

The final output represents:

```text
7 × 7 grid
15 predictions per grid cell
```

---

## Loss Function

The model has three different prediction tasks, so three different losses are used.

### 1. Objectness Loss

**BCEWithLogitsLoss**

Determines whether a grid cell contains an object.

```text
Object vs. No Object
```

### 2. Bounding Box Loss

**SmoothL1Loss**

Measures the difference between predicted and ground-truth bounding box values:

```text
x_offset
y_offset
width
height
```

Smooth L1 is used because it is less sensitive to very large errors than ordinary L1/L2 losses.

### 3. Classification Loss

**CrossEntropyLoss**

Determines which of the 10 Fashion-MNIST classes the detected object belongs to.

### Total Loss

The losses are combined as:

```text
Total Loss =
    Objectness Loss
    + 5 × Box Loss
    + 1 × Classification Loss
```

The bounding-box loss therefore receives a higher weight during training.

---

## Training Configuration

The main hyperparameters are:

| Parameter | Value |
|---|---:|
| Canvas size | 224×224 |
| Fashion-MNIST image size | 28×28 |
| Detection grid | 7×7 |
| Number of classes | 10 |
| Objects per image | 2–4 |
| Training samples | 10,000 |
| Validation samples | 2,000 |
| Batch size | 32 |
| Epochs | 15 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Box loss weight | 5.0 |
| Classification loss weight | 1.0 |
| Objectness threshold | 0.5 |
| Detection confidence threshold | 0.35 |
| IoU threshold | 0.5 |

A fixed random seed of `42` is used for reproducibility.

The model automatically uses CUDA when available:

```python
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise, training runs on the CPU.

---

## Training Process

For every training batch:

1. Synthetic images and targets are loaded.
2. Images are passed through the CNN.
3. The model produces predictions for every grid cell.
4. Objectness, bounding-box, and classification losses are calculated.
5. The individual losses are combined.
6. Gradients are calculated using backpropagation.
7. Adam updates the model parameters.

The training process records:

- Training loss
- Validation loss
- Objectness accuracy
- Classification accuracy
- Object recall
- Mean IoU
- Detection accuracy

---

## Validation Metrics

The model is evaluated without updating its weights.

### Objectness Accuracy

Measures how accurately the model determines whether each grid cell contains an object.

### Classification Accuracy

Measures how accurately detected objects are classified into the correct Fashion-MNIST category.

### Object Recall

Measures how many of the actual objects are successfully detected.

### Mean IoU

IoU, or **Intersection over Union**, measures the overlap between the predicted and ground-truth bounding boxes.

```text
IoU = Intersection Area / Union Area
```

An IoU of:

```text
1.0 → Perfect overlap
0.0 → No overlap
```

The project uses **0.5 IoU** as the reference threshold.

### Detection Accuracy

The notebook also tracks an overall detection accuracy based on its detection criteria.

---

## Bounding Box Decoding

During inference, the model predicts:

```text
x_offset
y_offset
width
height
```

These normalized predictions are converted back into actual pixel coordinates.

The grid cell location and predicted offsets are used to determine the object's center, after which the predicted width and height are used to calculate:

```text
x1
y1
x2
y2
```

The coordinates are clipped so that the predicted bounding box remains inside the 224×224 image.

---

## Inference

The `detect_objects()` function performs inference on a synthetic image.

For each grid cell:

1. Objectness logits are converted into probabilities using `sigmoid`.
2. Class logits are converted into class probabilities using `softmax`.
3. The most likely class is selected.
4. Objectness probability and class probability are multiplied to produce a confidence score.
5. Predictions below the confidence threshold are discarded.
6. Remaining bounding boxes are decoded into image coordinates.

Each detection contains:

```python
{
    "box": [...],
    "class_id": ...,
    "class_name": ...,
    "confidence": ...
}
```

---

## Visualization

The notebook visualizes both the ground truth and model predictions.

### Ground Truth

The synthetic image is displayed with the actual bounding boxes and class names.

### Model Predictions

The predicted bounding boxes are displayed together with:

```text
Class name
Confidence percentage
```

This makes it possible to visually inspect how well the detector is locating and classifying the objects.

---

## Training Curves

The notebook generates plots for:

### Training vs Validation Loss

Shows whether the model's loss decreases during training and how training performance compares with validation performance.

### Mean IoU

Tracks bounding-box localization performance across epochs.

The plot also displays the reference:

```text
IoU = 0.50
```

### Detection Accuracy

Shows how overall detection performance changes throughout training.

---

## Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **PIL / Pillow**
- **Matplotlib**

No OpenCV is used in this project.

---

## Project Structure

The main implementation is contained in the Jupyter notebook:

```text
imageclassification-without-opencv-2.ipynb
```

The notebook follows this general pipeline:

```text
Fashion-MNIST
      ↓
Synthetic Image Generation
      ↓
224×224 Canvas
      ↓
Ground-Truth Bounding Boxes
      ↓
7×7 Target Encoding
      ↓
CNN
      ↓
Objectness + Bounding Box + Classification
      ↓
Multi-task Loss
      ↓
Training
      ↓
Validation
      ↓
Object Detection
      ↓
Visualization
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy matplotlib pillow jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
imageclassification-without-opencv-2.ipynb
```

Run the notebook cells sequentially.

Fashion-MNIST will be downloaded automatically through `torchvision` when it is not already available locally.

---

## Key Learning Objectives

This project demonstrates the concepts behind building an object detector without relying on a pre-built detection model.

It covers:

- Fashion-MNIST data loading
- Synthetic object detection dataset generation
- Bounding box representation
- IoU calculation
- Grid-based object detection
- Target encoding
- CNN feature extraction
- Multi-task prediction
- Objectness prediction
- Bounding box regression
- Multi-class classification
- Custom loss functions
- Model training with Adam
- Validation metrics
- Bounding box decoding
- Confidence-based detection
- Prediction visualization

---

## Important Limitation

This project uses a **single-object prediction per grid cell**.

Each grid cell can store only one:

```text
objectness
bounding box
class
```

Therefore, if multiple object centers fall into the same grid cell, they cannot be independently represented by the target encoding.

The synthetic data generation and object placement strategy reduce this problem, but it remains an inherent limitation of the current grid representation.

---

## Future Improvements

Potential improvements include:

- Add multiple bounding-box predictions per grid cell
- Implement Non-Maximum Suppression (NMS)
- Add data augmentation
- Improve synthetic object placement
- Add more realistic object scaling
- Use a deeper feature extractor
- Add precision, recall, and F1-score
- Calculate mAP
- Compare performance against a pretrained detector
- Increase the synthetic dataset size
- Experiment with different grid sizes
- Tune objectness, box, and classification loss weights

---

## Summary

This project takes the original **Fashion-MNIST classification problem** and converts it into a **multi-object detection problem**.

A custom CNN learns to locate and classify multiple clothing items placed on a synthetic 224×224 image. The detector uses a **7×7 grid**, predicts objectness, bounding boxes, and class probabilities, and is trained using a combination of BCE, Smooth L1, and cross-entropy losses.

The project provides a compact implementation of the fundamental ideas behind **grid-based object detectors**, while keeping the entire pipeline understandable and implemented from the ground up.
