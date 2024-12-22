# بسم الله الرحمن الرحيم

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("A_Z Handwritten Data.csv", header=None)

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

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

print("################################## Starting First experiment... ######################################")

# Flatten the images for SVM input
images_flattened = images.reshape(images.shape[0], -1)

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    images_flattened, labels, test_size=0.3, random_state=42, stratify=labels)

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


def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{model_name}")
    print("Confusion Matrix:")
    print(cm)
    print(f"F1-Score: {f1}\n")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Evaluate linear SVM
evaluate_model(linear_svm, X_test, y_test, "Linear Kernel")

# Evaluate RBF SVM
evaluate_model(rbf_svm, X_test, y_test, "RBF Kernel")

print("################################## Starting Second experiment... ######################################")

images_flattened = images.reshape(images.shape[0], -1)

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.50, random_state=42)

class LogisticRegressionOVR:
    def __init__(self, num_classes, learning_rate=0.2, num_iterations=1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.biases = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((self.num_classes, num_features))
        self.biases = np.zeros(self.num_classes)

        y_one_hot = np.eye(self.num_classes)[y]

        for i in range(self.num_iterations):
            scores = np.dot(X, self.weights.T) + self.biases
            predictions = self.sigmoid(scores)

            error = predictions - y_one_hot
            gradient_weights = np.dot(error.T, X) / num_samples
            gradient_biases = np.sum(error, axis=0) / num_samples

            self.weights -= self.learning_rate * gradient_weights
            self.biases -= self.learning_rate * gradient_biases
            if (i + 1) % 100 == 0:
                print(f"End of Iteration {i+1}")

    def predict(self, X):
        scores = np.dot(X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)


# Train the model
num_classes = len(unique_classes)
model = LogisticRegressionOVR(num_classes=num_classes, learning_rate=0.2, num_iterations=1000)
print("Training logistic regression model...")
model.train(X_train_split, y_train_split)
print("Training complete!")

# Validate the model
val_predictions = model.predict(X_val)
val_f1 = f1_score(y_val, val_predictions, average='weighted')
print(f"Validation F1 Score: {val_f1:.4f}")


def plot_error_and_accuracy(y_true, predictions, title):
    cm = confusion_matrix(y_true, predictions)
    accuracy = np.trace(cm) / np.sum(cm)
    error = 1 - accuracy

    plt.figure()
    plt.bar(["Error", "Accuracy"], [error, accuracy], color=['red', 'green'])
    plt.title(title)
    plt.ylabel("Rate")
    plt.show()


plot_error_and_accuracy(y_val, val_predictions, "Validation Error and Accuracy")

# testing the model
evaluate_model(model, X_test, y_test, "Logistic Regression")

print("################################## Starting Third experiment... ######################################")


one_hot_labels = to_categorical(data.iloc[:, 0].values, num_classes=26)

X_train, X_temp, y_train, y_temp = train_test_split(images, one_hot_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model1 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(200, activation='relu'),
    Dense(26, activation='softmax')
])

model2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(300, activation='tanh'),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dense(26, activation='softmax')
])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history1 = model1.fit(X_train_split, y_train_split, validation_data=(X_val, y_val), epochs=15, batch_size=32, verbose=True)
history2 = model2.fit(X_train_split, y_train_split, validation_data=(X_val, y_val), epochs=15, batch_size=32, verbose=True)

# Plot accuracy and loss curves for both models
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.show()

plot_history(history1, "Model 1")
plot_history(history2, "Model 2")

# Save the best model
best_model = model2 if max(history2.history['val_accuracy']) > max(history1.history['val_accuracy']) else model1
worst_model = model1 if best_model == model2 else model2
best_model.save('best_model.h5')
worst_model.save('worst_model.h5')

if best_model == model1:
    print("Model 1 has the best accuracy")
else:
    print("Model 2 has the best accuracy")

# Reload and test the best model
loaded_model = load_model('best_model.h5')

# Evaluate on the test set
y_pred = np.argmax(loaded_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
plot_confusion_matrix(y_true, y_pred)

# F1 Score
f1 = f1_score(y_true, y_pred, average='macro')
print("F1 Score:", f1)
