import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Define your model as shown in the previous example

# Path to your dataset
data_dir = 'path/to/your/dataset'

# Get the list of all image files in the dataset
all_image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.jpg')]

# Create labels based on the assumption that positive examples are in a subdirectory named 'positive'
labels = [1 if 'positive' in file else 0 for file in all_image_files]

# Split the data into train, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(all_image_files, labels, test_size=0.2, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels, test_size=0.5, random_state=42)

# Create data generators for training, validation, and test
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',  # Use 'categorical' for multiclass tasks
    subset='training'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',  # Use 'categorical' for multiclass tasks
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'  # Use 'categorical' for multiclass tasks
)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks for training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks
)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test dataset
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)

# Print classification report
print(classification_report(test_generator.classes, y_pred))