from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

#dfdsafdsfsdffadsfdsffdasfd
# Define the ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Load a single image
img_path = '/home/b0da/Pictures/22.jpg'  # Update with the correct path to an image
img = image.load_img(img_path, target_size=(150, 150))  # Resize image
x = image.img_to_array(img)  # Convert image to numpy array
x = np.expand_dims(x, axis=0)  # Add a batch dimension

# Generate batches of augmented images
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:  # Display 4 augmented images
        break

plt.show()
