# بسم الله الرحمن الرحيم

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns

# change this file_path to match your dataset location
data_path = r"C:\Users\Beeko\A_Z Handwritten Data.csv"
print("Loading dataset...")
data = pd.read_csv(data_path, header=None)
print("Finished dataset loading")


print("################################## Data exploration and preparation ######################################")
# Extract labels and images
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values

unique_classes, class_count = np.unique(labels, return_counts=True)
print(f"Number of unique classes: {len(unique_classes)}")
print(f"Classes: {unique_classes}")
print(f"Class Counts: {class_count}")

# Plot the distribution
plt.figure(figsize=(10, 5))
plt.bar(unique_classes, class_count, color='blue')
plt.xlabel('Classes')
plt.ylabel('Number of images')
plt.title('Class distribution')
# plt.show() # uncomment if you want to see :)

# Reshape the image data to 28x28
images = images.reshape(-1, 28, 28)
print(f"Dataset loaded successfully, Total images: {len(images)}")

# Normalize the images
images = images / 255.0
print("Image data normalized successfully!")


unique_labels = np.unique(labels)
plt.figure()
for i, label in enumerate(unique_labels[:5]):  # Visualize first 5 unique classes
    idx = np.where(labels == label)[0][0]      # Get the first index of the label
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[idx], cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("images.png")                      # uncomment if you want to save :)

print("################################## Starting First experiment... ######################################")

# Flatten the images for SVM input
images_flattened = images.reshape(images.shape[0], -1)

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    images_flattened, labels, test_size=0.25, random_state=42, stratify=labels)

# Train linear SVM
print("Training linear SVM...")
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train, y_train)

# Train RBF SVM
print("Training RBF SVM...")
rbf_svm = SVC(kernel='rbf', random_state=42)
rbf_svm.fit(X_train, y_train)

# Evaluate models
print("Evaluating models...")


def evaluate_model(model, x_test, y_test, kernel_name):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{kernel_name} Kernel")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"F1-Score: {f1}\n")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix ({kernel_name} Kernel)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Evaluate linear SVM
evaluate_model(linear_svm, X_test, y_test, "Linear")

# Evaluate RBF SVM
evaluate_model(rbf_svm, X_test, y_test, "RBF")
