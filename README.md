import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# Define parameters
input_shape = (256, 256, 3)  # Input image shape
latent_dim = 128  # Dimension of the latent space
num_samples = 10000  # Number of samples for training

# Define generator model
def build_generator(latent_dim, input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64 * 64 * 128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((64, 64, 128)))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(3, (4,4), activation='sigmoid', padding='same'))
    model.summary()
    noise = layers.Input(shape=(latent_dim,))
    img = model(noise)
    return models.Model(noise, img)

# Load and preprocess product images
def load_product_images(file_paths):
    images = []
    for path in file_paths:
        img = Image.open(path)
        img = img.resize((256, 256))
        img = np.array(img) / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Train the generator model
def train_generator(generator, product_images, epochs=100, batch_size=32):
    for epoch in range(epochs):
        idx = np.random.randint(0, product_images.shape[0], batch_size)
        real_images = product_images[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        # Update generator weights using adversarial loss or any other suitable loss function
        # Here, we'll use a simple MSE loss as an example
        loss = tf.keras.losses.mean_squared_error(real_images, fake_images)
        generator.train_on_batch(noise, real_images)
        print(f"Epoch {epoch+1}, Loss: {np.mean(loss)}")

# Generate advertisement images
def generate_advertisement_images(generator, product_images, num_samples=10):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_images = generator.predict(noise)
    # Combine product images and generated images, add text overlays or prompts, etc.
    # Return the final advertisement images
    return generated_images
    
