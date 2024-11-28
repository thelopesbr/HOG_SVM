# HOG + SVM for Binary Image Classification

This project addresses a binary image classification task: identifying the presence or absence of pedestrians in images. The goal is to build and evaluate a classification model capable of accurately distinguishing between images containing pedestrians (positive examples) and those that do not (negative examples).

---

## Dataset

The **INRIAPerson** dataset was used, containing **2487 images**:
- **1269 images with pedestrians**
- **1218 images without pedestrians**

The dataset was sourced from the Hugging Face platform, provided by [marcelarosalesj](https://huggingface.co/datasets/marcelarosalesj/inria-person/tree/main).

### Preprocessing
1. **Resizing**: Images without pedestrians were resized to match the dimensions of the pedestrian images (**96x160 pixels**) using the Pillow (PIL) library.
2. **Balancing Classes**: The dataset was balanced by equalizing the number of images in each class. For certain tests, only **100 images per class** were used to evaluate the model's performance with a smaller dataset.
3. **Splitting Data**: The dataset was divided into training and testing sets with a **70-30 split**, ensuring both sets were balanced.

---

## Methodology

### Feature Extraction
- **Histogram of Oriented Gradients (HOG)** was used to extract features from the images.
- Parameters for HOG:
  - `pixels_per_cell=(8, 8)`
  - `cells_per_block=(2, 2)`
  - `block_norm='L2-Hys'`
- All images were converted to grayscale using `rgb2gray` from the `scikit-image` library before applying HOG.

### Model Training
- A **Support Vector Machine (SVM)** classifier was used for training, with hyperparameter tuning performed using **RandomizedSearchCV**.
- Hyperparameters optimized:
  - `C` (Regularization): Tested over a continuous range from 0.001 to 1000.
  - `kernel`: Evaluated `linear`, `rbf`, `poly`, and `sigmoid` kernels.
  - `gamma`: Sampled from a range of 0.001 to 10.
  - `degree`: Tested for polynomial kernels (values 2-5).
  - `class_weight`: `balanced` and `None`.

### Evaluation
- The model's performance was evaluated on the test set using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

---

## Results

### Main Test
Using the full dataset:
- **Training Set**: 1705 images
- **Test Set**: 731 images
- **Best Parameters**: 
  - `C`: 596.85
  - `kernel`: Polynomial
  - `degree`: 4
  - `gamma`: 1.0
  - `class_weight`: None
- **Performance**:
  - **Training Accuracy**: 100% (0.00 error rate)
  - **Test Accuracy**: 98% (0.02 error rate)
  - **Class 0 (No Pedestrians)**: Precision = 97%, Recall = 98%, F1-score = 98%
  - **Class 1 (Pedestrians)**: Precision = 98%, Recall = 97%, F1-score = 98%

### Additional Test
Using only **100 images per class**:
- **Training Set**: 140 images (70 per class)
- **Test Set**: 60 images (30 per class)
- **Best Parameters**:
  - `C`: 374.54
  - `kernel`: Linear
  - `class_weight`: Balanced
- **Performance**:
  - **Training Accuracy**: 100% (0.00 error rate)
  - **Test Accuracy**: 100% (0.00 error rate)

---

## Conclusion

The combination of HOG features and SVM provided highly accurate and reliable results for this binary image classification task. The model demonstrated exceptional performance across both the full dataset and a reduced dataset of 100 images per class, achieving near-perfect accuracy. This study highlights the effectiveness of using HOG for feature extraction and SVM for classification, especially with appropriate hyperparameter tuning.

Further testing on more diverse datasets is recommended to validate the model's robustness in real-world scenarios.

---

## How to Use

1. Clone the repository.
2. Ensure the required libraries are installed (`scikit-learn`, `scikit-image`, `Pillow`, etc.).
3. Run the provided scripts to reproduce the preprocessing, feature extraction, training, and evaluation steps.
4. Refer to the results and additional test scripts for further experimentation.

---

## References

- Dataset: [INRIAPerson on Hugging Face](https://huggingface.co/datasets/marcelarosalesj/inria-person/tree/main)
- Libraries: `scikit-learn`, `scikit-image`, `Pillow`

---

Feel free to explore the code and adapt it for similar binary classification tasks!
