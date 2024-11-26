from keras.applications import VGG16

# Load the VGG16 model, excluding the top fully connected layers
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Print the model summary
print(conv_base.summary())