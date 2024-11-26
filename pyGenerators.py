import numpy as np
from tensorflow.keras.datasets import mnist

def mnist_data_generator(X, y, batch_size):
    """
    Generator function to yield batches of MNIST data.
    
    Args:
        X: numpy array, input data (images).
        y: numpy array, labels corresponding to the input data.
        batch_size: int, the size of each batch.
    
    Yields:
        Batches of (X_batch, y_batch) of the specified batch size.
    """
    num_samples = len(X)
    
    # Shuffle the data at the start of each epoch
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        yield X_batch, y_batch

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the image data to [0, 1] range
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Set the batch size
batch_size = 64

# Create the generator
generator = mnist_data_generator(X_train, y_train, batch_size)

# Example usage: Fetch 3 batches
if __name__ == "__main__":
    for i, (X_batch, y_batch) in enumerate(generator):
        print(f"Batch {i+1}")
        print(f"X_batch shape: {X_batch.shape}")  # Shape should be (batch_size, 28, 28)
        print(f"y_batch shape: {y_batch.shape}")  # Shape should be (batch_size,)
        
        # Stop after 3 batches
        if i == 2:
            break
