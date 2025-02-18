# 🌍 Intel Image Classification using CNNs in PyTorch

## 📌 Project Overview
This project builds a **Convolutional Neural Network (CNN)** from scratch to classify images into six categories:

🏢 **Buildings** | 🌲 **Forest** | ❄️ **Glacier** | ⛰ **Mountain** | 🌊 **Sea** | 🚦 **Street**

Using **PyTorch**, we train the CNN on the **Intel Image Classification Dataset**, optimizing it for accuracy and generalization. The final model achieves a **test accuracy of 84.13%**.

## 🔗 View the Notebook on Kaggle  
You can view and run this notebook directly on **Kaggle** using the link below:  

➡️ **[Intel Image Classifier - Kaggle Notebook](https://www.kaggle.com/code/andrslvarezpea/image-classifier-with-cnns-in-pytorch)**

---

## 📂 Dataset
The dataset contains images of **natural scenes** categorized into six classes. It is divided with one folder for training and another one for testing


### 🔹 Preprocessing Steps:
- **Resize** all images to `150x150 pixels` for consistency.
- **Normalize pixel values** to `[-1, 1]` for stable training.
- **Split the dataset** into **80% training / 20% validation** for better evaluation.

---

## 🛠 Model Architecture
We designed a **custom CNN** with the following structure:
- **3 Convolutional layers** (ReLU activation, MaxPooling)
- **Fully connected layers** with Dropout for regularization
- **CrossEntropy Loss** (for multi-class classification)
- **Adam Optimizer** with an initial learning rate of `0.001`

---

## 🚀 Training Process
The model is trained using:
- **Mixed Precision Training (AMP)** for faster computations
- **Batch Size of 64** for stable gradient updates
- **Validation Monitoring** (to track overfitting)

### 🔹 Training Configuration:
- **First Training:** Ran for `10 epochs`, but validation loss started fluctuating → Overfitting detected.
- **Final Training:** Reduced to `5 epochs`, achieving an optimal balance of accuracy and efficiency.

---

## 📊 Results
| **Epochs** | **Validation Accuracy** | **Validation Loss** |
|------------|---------------------|-----------------|
| 10 (Initial) | 83.61% | 0.7473 |
| **5 (Final)** | **84.04%** | **0.5058** |

### 🔹 Test Accuracy: **84.13%**

---

## 🔍 Conclusion & Future Improvements
While the CNN performs well, **further improvements could be made**:
- **Hyperparameter Tuning**: Adjust learning rates, batch sizes, and optimizer settings.
- **Transfer Learning**: Use a **pretrained model (ResNet18, MobileNetV2)** to improve accuracy.
- **Data Augmentation**: Add **random rotation, flipping, and cropping** for better generalization.
- **Regularization Techniques**: Increase **dropout, L2 weight decay, or batch normalization**.

By implementing these techniques, we could push the model **beyond 84% accuracy** while improving efficiency. 🚀
