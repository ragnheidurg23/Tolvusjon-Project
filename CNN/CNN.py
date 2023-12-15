from PIL import Image
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


# Path to your dataset
data_dir = 'CNN\Original Images'
batch_size = 32
epochs = 10

# Define image dimensions
image_height = 640  # Replace with your desired height
image_width = 1280   # Replace with your desired width

# Read CSV file without header
csv_path = 'CNN\Images with Meiboscore - Binary.csv'  # Replace with the actual path
df = pd.read_csv(csv_path, header=None, names=['ImageFileNameColumn', 'LabelColumn'])

# Display the DataFrame (for verification)
print(df.head())

# Get the list of all image files in the dataset
all_image_files = df['ImageFileNameColumn']
data_size = len(all_image_files)

# Create an empty array to store the images
all_images = np.zeros((data_size, image_height, image_width, 3))  # Assuming RGB images

# Load images into the array
for i, file_path in enumerate(all_image_files):
    # Construct the full path to the image
    full_path = os.path.join(data_dir, file_path)

    # Read and resize the image
    img = Image.open(full_path)
    img = img.resize((image_width, image_height))
    img_array = np.array(img)

    # Normalize the pixel values
    img_array = img_array / 255.0

    # Store the image in the array
    all_images[i] = img_array

# Create labels based on the assumption that positive examples are in a subdirectory named 'positive'
labels = df['LabelColumn']

# Split the data into train, validation, and test sets
train_files, test_val_files, train_labels, test_val_labels = train_test_split(all_images, labels, test_size=0.2, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(test_val_files, test_val_labels, test_size=0.5, random_state=42)

# Create data generators for training, validation, and test
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_files, train_labels, batch_size=batch_size)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow(val_files, val_labels, batch_size=batch_size)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow(test_files, test_labels, batch_size=batch_size)

# Define a simple CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer to transition from convolutional layers to fully connected layers
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout for regularization
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification, use 'softmax' for multiclass tasks

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks for training
callbacks = [
    ModelCheckpoint(filepath='best_model.h5', save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True),
    TensorBoard(log_dir='./logs')
]

# Train the model with a progress bar
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=1,  # Train for one epoch
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Calculate and print metrics
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validation Accuracy: {val_accuracy}")

# Evaluate the model on the test dataset
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)

# Print classification report
print(classification_report(test_generator.classes, y_pred))