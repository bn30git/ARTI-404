import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import os

DATASET_PATH = 'C:/Users/user/Downloads/ARC/arcDataset/'
PROCESSED_DATASET_PATH = 'C:/Users/user/Downloads/ARC/PrData2/'

classes = {'Art Deco architecture': 0, 'Greek Revival architecture': 1}
X = []
Y = []

target_size = (600, 600)

def process_image(image):
    # Image resizing with padding
    resized_image = cv2.resize(image, target_size)
    if resized_image.shape != target_size:
        pad_h = target_size[0] - resized_image.shape[0]
        pad_w = target_size[1] - resized_image.shape[1]
        resized_image = np.pad(resized_image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

        # Enhance Brightness
        from PIL import Image
        pil_image = Image.fromarray(resized_image)
        enhancer = ImageEnhance.Brightness(pil_image)
        bright_image = enhancer.enhance(0.5)

        # Enhance Contrast
        enhancer = ImageEnhance.Contrast(bright_image)
        contrast_image = enhancer.enhance(6)

        # Enhance Sharpness
        enhancer = ImageEnhance.Sharpness(contrast_image)
        sharp_image = enhancer.enhance(6)
        enhanced_array = np.array(sharp_image)

        # Edge detection
        # Prewitt Operator
        gray = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2GRAY)
        prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        # Apply Prewitt Operator
        edge_x = cv2.filter2D(gray, -1, prewitt_x)
        edge_y = cv2.filter2D(gray, -1, prewitt_y)
        both = edge_y + edge_x
        # Data Augmentation (Horizontal Flip)
        if np.random.random() > 0.2:  # Apply flip with 20% probability
            both = cv2.flip(both, 1)
        return both


for cls in classes:
    pth = DATASET_PATH + cls
    for j in os.listdir(pth):
        image_path = pth + '/' + j
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processed_image = process_image(image)

        # Save the images
        output_dir = PROCESSED_DATASET_PATH + cls
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(os.path.join(output_dir, f'{base_filename}_processed.jpg'), processed_image)

        # Prepare for the model
        X.append(processed_image)
        Y.append(classes[cls])

# Data preparation and splitting
X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)
X_train, X_test, Y_train, Y_test = train_test_split(X_updated, Y, random_state=10, test_size=0.20)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# Reshape to include the image dimensions
X_train_scaled = X_train_scaled.reshape(-1, 600, 600, 1)
X_test_scaled = X_test_scaled.reshape(-1, 600, 600, 1)

#-------Building CNN-------
num_of_classes = 2  # Update with the number of architecture classes
model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(600, 600, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # For regularization
model.add(layers.Dense(num_of_classes, activation='sigmoid'))

#Compile NN
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['acc'])

#Train NN
history = model.fit(X_train_scaled, Y_train, validation_split=0.1,epochs=10)

loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Loss =', loss)
print('Test Accuracy =', accuracy)

#------Classification Report------
Ypred = model.predict(X_test_scaled)
ypred = np.argmax(Ypred, axis=1)
print(classification_report(Y_test, ypred))

#Plot confusion matrix
cf_matrix = confusion_matrix(Y_test, ypred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.savefig('CNN_ConfusionMatrix.png')
plt.show()