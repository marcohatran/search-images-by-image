from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

# Prepare output file
output = open("indexing.csv", "w")

# Loading pre-trained model
model = VGG16(weights='imagenet', include_top=False)

# Loading folder contains data inside
location = './dataset/indexing'  # Root folder
directory = os.listdir(location)
directory = sorted(directory)  # Arrange the data
for i in directory:
    img = image.load_img(location + '/' + i, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # Extract image feature by using convolutional layers
    vgg16_feature = model.predict(img_data)
    features = np.array(vgg16_feature).flatten()

    # Format output and store in DB
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (i, ",".join(features)))
    
    