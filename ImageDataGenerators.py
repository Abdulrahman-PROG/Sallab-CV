from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Update the path for Kaggle datasets
train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/dogs-vs-cats/train',  # Update this path with your dataset's train directory
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    '/kaggle/input/dogs-vs-cats/validation',  # Update this path with your dataset's validation directory
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
