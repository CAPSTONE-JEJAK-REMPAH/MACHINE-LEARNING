import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# Define spice categories
spice_categories = [
    'adas', 'andaliman', 'asam jawa', 'bawang bombai', 'bawang merah', 'bawang putih', 'biji ketumbar',
    'bukan rempah', 'bunga lawang', 'cengkeh', 'daun jeruk', 'daun kemangi', 'daun ketumbar', 'daun salam', 
    'jahe', 'jinten', 'kapulaga', 'kayu manis', 'kayu secang', 'kemiri', 'kemukus', 'kencur', 'kluwek', 
    'kunyit', 'lada', 'lengkuas', 'pala', 'saffron', 'serai', 'vanili', 'wijen'
]

# Load the saved model
model = tf.keras.models.load_model('C:/Users/fadhi/CAPSTONE_PROJECT/Spicesnew.h5')

# Load and preprocess the test image
img_path = "C:/Users/fadhi/CAPSTONE_PROJECT/From Google for Testing/112a2ec178808b0eadcfc003633071.jpg"
test_image = image.load_img(img_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)

# Normalize the image (rescale pixel values to [0, 1])
test_image = test_image / 255.0

# Expand dimensions to match the model input (batch size of 1)
test_image = np.expand_dims(test_image, axis=0)

# Predict the category of an image
result = model.predict(test_image)

# Extract the index with the highest probability
predicted_index = np.argmax(result[0])

# Fetch the corresponding spice category
predicted_category = spice_categories[predicted_index]
print(f'*** The spice is: {predicted_category}')
