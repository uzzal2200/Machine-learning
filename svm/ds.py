import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to load .gif images
def load_gif_images(folder_path, image_size=(64, 64)):
    images = []
    labels = []
    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)
        if os.path.isdir(emotion_path):  # Ensure it's a directory
            for filename in os.listdir(emotion_path):
                if filename.endswith('.gif'):
                    gif_path = os.path.join(emotion_path, filename)
                    with Image.open(gif_path) as gif:
                        frame = gif.convert('L').resize(image_size)
                        images.append(np.array(frame))
                        labels.append(emotion)  # Use the folder name as the label
    return np.array(images), np.array(labels)

# Paths
dataset_path = r"D:\batch -3\Machine Learning\svm\new"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Load datasets
X_train, y_train = load_gif_images(train_path)
X_test, y_test = load_gif_images(test_path)

# Encode labels into numeric format
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Flatten images for SVM input
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten to (n_samples, 4096)

# Check the shapes
print(f"Shape of X_train_flat: {X_train_flat.shape}")  # Should be (n_samples, 4096)
print(f"Shape of y_train_encoded: {y_train_encoded.shape}")  # Should be (n_samples,)

# Train an SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_flat, y_train_encoded)

# Predict on test data
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_pred = svm.predict(X_test_flat)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=encoder.classes_))

print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred):.2f}")
