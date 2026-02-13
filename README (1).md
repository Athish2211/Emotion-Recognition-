# 🎭 Emotion Detection using Deep Learning

A Convolutional Neural Network (CNN) based emotion recognition system that classifies facial expressions into 7 different emotions using TensorFlow and Keras.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a deep learning model for facial emotion recognition, capable of classifying images into 7 distinct emotional categories:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😮 Surprise

The model is trained on grayscale facial images (48x48 pixels) and achieves robust performance through data augmentation and regularization techniques.

## ✨ Features

- **Multi-class Classification**: Recognizes 7 different emotional states
- **Data Augmentation**: Implements rotation, zoom, and horizontal flip for better generalization
- **Regularization**: Uses Dropout and Batch Normalization to prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction for optimal convergence
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Comprehensive Visualization**: Includes training metrics, confusion matrix, and sample predictions
- **Model Persistence**: Saves trained model for deployment

## 📊 Dataset

The project uses an emotion dataset with the following structure:
```
images/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

**Image Specifications:**
- Format: Grayscale
- Size: 48x48 pixels
- Classes: 7 emotions

## 🏗️ Model Architecture

The CNN model consists of the following layers:

```
Input (48x48x1)
    ↓
Conv2D (64 filters, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling2D (2x2) + Dropout (0.25)
    ↓
Conv2D (128 filters, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling2D (2x2) + Dropout (0.25)
    ↓
Conv2D (256 filters, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling2D (2x2) + Dropout (0.25)
    ↓
Flatten
    ↓
Dense (512) + ReLU + BatchNorm + Dropout (0.5)
    ↓
Dense (7) + Softmax
```

**Key Components:**
- **Convolutional Layers**: Extract spatial features from images
- **Batch Normalization**: Stabilizes and accelerates training
- **Dropout**: Reduces overfitting (25% in conv layers, 50% in dense layer)
- **Activation**: ReLU for hidden layers, Softmax for output

## 🚀 Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- CUDA (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-detection-dl.git
cd emotion-detection-dl
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `requirements.txt` with the following dependencies:
```txt
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=1.0.0
```

## 💻 Usage

### Training the Model

1. **Prepare your dataset**: Organize images in the directory structure shown above

2. **Run Exploratory Data Analysis**:
```bash
jupyter notebook DL_Hackathon_EDA.ipynb
```

3. **Train the model**:
```bash
jupyter notebook DL_Hackathon_Model_Training_and_Performance_Metrics.ipynb
```

Or run as Python script:
```python
# Update data paths in the notebook to match your directory
data_dir = "path/to/your/images"
train_dir = "path/to/your/images/train"
validation_dir = "path/to/your/images/validation"
```

### Making Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('emotion_detection.h5')

# Define emotion classes
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Predict emotion from image
def detect_emotion(image_path):
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)
    
    return predicted_class, confidence

# Example usage
emotion, conf = detect_emotion('path/to/test/image.jpg')
print(f"Predicted Emotion: {emotion} ({conf}% confidence)")
```

## 📈 Results

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)
- **Callbacks**: 
  - Early Stopping (patience: 5)
  - Learning Rate Reduction (factor: 0.2, patience: 3)

### Data Augmentation
- Rescaling: 1./255
- Rotation Range: 20 degrees
- Zoom Range: 0.2
- Horizontal Flip: True

### Performance Metrics
The model generates:
- Training/Validation Accuracy curves
- Training/Validation Loss curves
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## 📁 Project Structure

```
emotion-detection-dl/
│
├── DL_Hackathon_EDA.ipynb                          # Exploratory Data Analysis
├── DL_Hackathon_Model_Training_and_Performance_Metrics.ipynb  # Model Training
├── emotion_detection.h5                             # Saved model (after training)
├── requirements.txt                                 # Python dependencies
├── README.md                                        # Project documentation
│
└── images/                                          # Dataset directory
    ├── train/                                       # Training images
    └── validation/                                  # Validation images
```

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Metrics and evaluation

## 📊 Exploratory Data Analysis

The EDA notebook includes:
- Dataset statistics and class distribution
- Visualization of emotion class counts
- Pie charts showing data distribution
- Sample image visualization
- Train/validation split analysis

## 🎯 Key Features of Implementation

1. **Robust Data Pipeline**: Uses `ImageDataGenerator` for efficient batch processing
2. **Regularization Strategy**: Multiple dropout layers and batch normalization
3. **Adaptive Learning**: Learning rate scheduling for optimal convergence
4. **Model Checkpointing**: Saves best model based on validation performance
5. **Comprehensive Evaluation**: Detailed classification metrics and confusion matrix

## 🔮 Future Enhancements

- [ ] Real-time emotion detection from webcam
- [ ] Model optimization for mobile deployment
- [ ] Transfer learning with pre-trained models (VGG, ResNet)
- [ ] Multi-face detection in single image
- [ ] Web application interface
- [ ] REST API for emotion detection service

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- Dataset providers for emotion recognition dataset
- TensorFlow and Keras communities
- Deep Learning Hackathon organizers

## 📧 Contact

Your Name - [@yourhandle](https://twitter.com/yourhandle)

Project Link: [https://github.com/yourusername/emotion-detection-dl](https://github.com/yourusername/emotion-detection-dl)

---

**Note**: Update the dataset paths in the notebooks to match your local directory structure before running.
