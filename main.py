
import os
from PIL import Image

# Organizing the dataset
def check_image_dimensions(folder):
    # List to store the dimensions of each image
    dimensions = []
    
    # Iterate through all files in the folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):  # Check if it is a file
            try:
                with Image.open(file_path) as img:
                    dimensions.append(img.size)  # Add dimensions (width, height)
            except Exception as e:
                print(f"Error opening file {file}: {e}")
    return len(set(dimensions)) == 1

def resize_images(source_folder, destination_folder, width=96, height=160):
    # Check if the destination folder exists, otherwise create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file in os.listdir(source_folder):
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)

        # Check if the file is an image
        if os.path.isfile(source_path):
            try:
                with Image.open(source_path) as img:
                    # Resize the image
                    resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                    # Save to the destination folder
                    resized_img.save(destination_path)
            except Exception as e:
                print(f"Error processing image '{file}': {e}")

    print(f"Images resized and saved in '{destination_folder}'.")

def process_pedestrian_data(pedestrians, no_pedestrians, folder_ped="data_ped/pedestrians", folder_no_ped="data_ped/no_pedestrians"):
    """
    Handles the resizing of images based on the presence of pedestrians.

    Args:
        pedestrians (bool): Indicates if pedestrian images need to be processed.
        no_pedestrians (bool): Indicates if non-pedestrian images need to be processed.

    Returns:
        tuple: Paths to the folders containing resized pedestrian and non-pedestrian images.
    """

    if not pedestrians and not os.path.exists(f'{folder_ped}_resized'):
        print("Resizing with pedestrians...")
        aux = folder_ped + "_resized"
        os.makedirs(folder_ped, exist_ok=True)
        resize_images(folder_ped, aux, width=96, height=160)
        folder_ped = aux
    if not no_pedestrians and not os.path.exists(f'{folder_no_ped}_resized'):
        print("Resizing without pedestrians...")
        aux =  folder_no_ped + "_resized"
        os.makedirs(folder_no_ped, exist_ok=True)
        resize_images(folder_no_ped, aux, width=96, height=160)
        folder_no_ped = aux

    if os.path.exists(f'{folder_ped}_resized'):
        folder_ped = f'{folder_ped}_resized'
    if os.path.exists(f'{folder_no_ped}_resized'):
        folder_no_ped = f'{folder_no_ped}_resized'

    return folder_ped, folder_no_ped

def equalize_image_count(folder1, folder2):
    """
    Ensures that both folders have the same number of images by removing extra files
    from the folder with more images.

    Args:
        folder1 (str): Path to the first folder.
        folder2 (str): Path to the second folder.

    Returns:
        None
    """
    # List images in both folders
    images1 = sorted(os.listdir(folder1))  # Sort to ensure consistency
    images2 = sorted(os.listdir(folder2))
    
    count1 = len(images1)
    count2 = len(images2)

    if count1 == count2:
        print("Both folders have the same number of images.")
        return
    elif count1 > count2:
        excess_files = images1[count2:]  # Extra files in folder1
        for file in excess_files:
            file_path = os.path.join(folder1, file)
            os.remove(file_path)

    elif count2 > count1:
        excess_files = images2[count1:]  # Extra files in folder2
        for file in excess_files:
            file_path = os.path.join(folder2, file)
            os.remove(file_path)
           
    
    print(f"Equalized image count: {min(count1, count2)} images in each folder.")

def limit_image_count(folder, limit):
    """
    Ensures that a folder contains no more than the specified number of images by removing extra files.

    Args:
        folder (str): Path to the folder.
        limit (int): Maximum number of images to retain in the folder.

    Returns:
        None
    """
    # List and sort images in the folder
    images = sorted(os.listdir(folder))  # Sort to ensure consistency
    count = len(images)

    if count <= limit:
        print(f"The folder '{folder}' already contains {count} images, which is within the limit of {limit}.")
        return

    # Remove excess files
    excess_files = images[limit:]  # Files beyond the limit
    for file in excess_files:
        file_path = os.path.join(folder, file)
        os.remove(file_path)
    
    print(f"The folder '{folder}' now contains {limit} images.")

folder_ped = "data_ped/pedestrians"
folder_no_ped = "data_ped/no_pedestrians"

pedestrians = check_image_dimensions(folder_ped)
no_pedestrians = check_image_dimensions(folder_no_ped)

folder_ped, folder_no_ped = process_pedestrian_data(pedestrians, no_pedestrians, folder_ped, folder_no_ped)

#Use to equalize based on the class with the least amount of images
equalize_image_count(folder_ped, folder_no_ped)

# or use to limit the number of images in each class

# limit_image_count(folder_ped, 100)
# limit_image_count(folder_no_ped, 100)

print(f"Images with pedestrians: {len(os.listdir(folder_ped))}")
print(f"Images without pedestrians: {len(os.listdir(folder_no_ped))}")
print()

# separating into training and testing and calculating HOG
# -----------------------------------------------------------

from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np

def load_images_from_folder(folder):
    """
    Load image file paths from a given folder.

    Args:
        folder (str): Path to the folder.

    Returns:
        list: List of image file paths.
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def calculate_hog(image_path):
    """
    Calculate the HOG features for an image.

    Args:
        image_path (str): Path to the image.

    Returns:
        np.array: HOG feature vector.
    """
    image = imread(image_path)
    if len(image.shape) > 2:  # Convert to grayscale if it's an RGB image
        image = rgb2gray(image)
    features, _ = hog(image, 
                      pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), 
                      block_norm='L2-Hys', 
                      visualize=True)
    return np.array(features)

def split_and_balance(pedestrian_images, no_pedestrian_images, test_size=0.3):
    """
    Split data into balanced training and testing sets using stratified sampling.

    Args:
        pedestrian_images (list): List of pedestrian image paths.
        no_pedestrian_images (list): List of non-pedestrian image paths.
        test_size (float): Proportion of data to allocate to the test set.

    Returns:
        tuple: Training and test sets (X_train, X_test, y_train, y_test).
    """
    # Combine all data and labels
    X = pedestrian_images + no_pedestrian_images
    y = [1] * len(pedestrian_images) + [0] * len(no_pedestrian_images)

    # Split the data, using stratify to balance classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def process_and_extract_features(X_train, X_test):
    """
    Extract HOG features for training and testing sets.

    Args:
        X_train (list): List of training image paths.
        X_test (list): List of testing image paths.

    Returns:
        tuple: HOG feature matrices for training and testing sets.
    """
    X_train_hog = [calculate_hog(img_path) for img_path in X_train]
    X_test_hog = [calculate_hog(img_path) for img_path in X_test]

    # Convert to consistent NumPy arrays with padding/truncation if needed
    max_length = max(max(len(x) for x in X_train_hog), max(len(x) for x in X_test_hog))
    X_train_hog = np.array([np.pad(x, (0, max_length - len(x))) for x in X_train_hog])
    X_test_hog = np.array([np.pad(x, (0, max_length - len(x))) for x in X_test_hog])

    return X_train_hog, X_test_hog

pedestrian_images = load_images_from_folder(folder_ped)
no_pedestrian_images = load_images_from_folder(folder_no_ped)

# Split and balance the data
X_train, X_test, y_train, y_test = split_and_balance(pedestrian_images, no_pedestrian_images, test_size=0.3)

# Extract HOG features
X_train_hog, X_test_hog = process_and_extract_features(X_train, X_test)

# Final outputs
print(f"Training set size: {len(X_train_hog)}")
print(f"Test set size: {len(X_test_hog)}")
print(f"Feature vector length (HOG): {X_train_hog[0].shape}")

# Tuning SVM hyperparameters
# -----------------------------------------------------------

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform

def train_svm_with_extensive_search(X_train, y_train, X_test, y_test, n_iter=50):
    """
    Train an SVM model with extensive hyperparameter tuning using RandomizedSearchCV.

    Args:
        X_train (np.array): HOG feature vectors for training.
        y_train (list): Labels for training data.
        X_test (np.array): HOG feature vectors for testing.
        y_test (list): Labels for testing data.
        n_iter (int): Number of hyperparameter combinations to test.

    Returns:
        None
    """
    # Define the SVM model
    svm = SVC()

    # Define the hyperparameter distribution
    param_distributions = {
        'C': uniform(0.001, 1000),  # Continuous range for C
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Include more kernels
        'gamma': uniform(0.001, 10),  # Range for gamma
        'degree': [2, 3, 4, 5],  # Polynomial degrees (only used for 'poly' kernel)
        'class_weight': ['balanced', None],  # Handle class imbalance
    }

    # Perform randomized search
    random_search = RandomizedSearchCV(
        svm,
        param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)

    # Best parameters and best estimator
    print("Best Parameters:", random_search.best_params_)
    best_model = random_search.best_estimator_

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return best_model


best_model = train_svm_with_extensive_search(X_train_hog, y_train, X_test_hog, y_test, n_iter=50)

# Classifying test images
# -----------------------------------------------------------
import os
from sklearn.metrics import classification_report, accuracy_score
from shutil import copy2

def classify_test_images(model, X_test, y_test, test_image_paths, output_folder="classified_images"):
    """
    Classify test images using a trained SVM model and save results in corresponding folders.

    Args:
        model: Trained SVM model.
        X_test (np.array): HOG feature vectors for testing.
        y_test (list): True labels for testing.
        test_image_paths (list): Paths to the test images.
        output_folder (str): Folder to save classified images into subfolders by class.

    Returns:
        None
    """
    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Create output subfolders for each class
    class_folders = {0: os.path.join(output_folder, "class_0"), 
                     1: os.path.join(output_folder, "class_1")}
    for folder in class_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save classified images to the respective subfolders
    for img_path, pred_label in zip(test_image_paths, y_pred):
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(class_folders[pred_label], img_name)
        copy2(img_path, dest_path)  # Move the file to the appropriate folder

classify_test_images(
    model=best_model,  # O modelo SVM treinado
    X_test=X_test_hog,  # Vetores de características HOG do conjunto de teste
    y_test=y_test,  # Rótulos verdadeiros do conjunto de teste
    test_image_paths=X_test,  # Caminhos das imagens do conjunto de teste
    output_folder="classified_images"
)

# Calculating accuracy
# -----------------------------------------------------------

def compute_accuracy(model, X_train, y_train, X_test, y_test):
    """
    Compute the accuracy and error rate for the training and test datasets.

    Args:
        model: Trained SVM model.
        X_train (np.array): HOG feature vectors for training.
        y_train (list): Labels for training data.
        X_test (np.array): HOG feature vectors for testing.
        y_test (list): Labels for testing data.

    Returns:
        None
    """
    # Predictions on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_error = 1 - train_accuracy

    # Predictions on test data
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_error = 1 - test_accuracy

    # Print results
    print(f"Training Accuracy: {train_accuracy:.2f} ({train_error:.2f} error rate)")
    print(f"Test Accuracy: {test_accuracy:.2f} ({test_error:.2f} error rate)")



compute_accuracy(best_model, X_train_hog, y_train, X_test_hog, y_test)