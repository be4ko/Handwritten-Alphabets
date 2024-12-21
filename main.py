# بسم الله الرحمن الرحيم

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# change this file_path to match your dataset location
data_path = r"C:\Users\Beeko\A_Z Handwritten Data.csv"
print("Loading dataset...")
data = pd.read_csv(data_path, header=None)
print("Finished dataset loading")


print("################################## Data exploration and preparation ######################################")
# Extract labels and images
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values

unique_classes, class_counts = np.unique(labels, return_counts=True)
print(f"Number of unique classes: {len(unique_classes)}")
print(f"Classes: {unique_classes}")
print(f"Class Counts: {class_counts}")

# Plot the distribution
plt.figure(figsize=(10, 5))
plt.bar(unique_classes, class_counts, color='blue')
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
# plt.savefig("images.png") # uncomment if you want to save :)

print("################################## First experiment  ######################################")
