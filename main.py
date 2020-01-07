from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import csv
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Compare similarity of cosine algorithm between 2 vectors
def compareCosineSimilarity(feature1, feature2):
    f1 = np.array(feature1)
    f2 = np.array(feature2)
    aa = f1.reshape(1,f1.shape[0])
    ba = f2.reshape(1,f2.shape[0])
    cos_lib = cosine_similarity(aa, ba)
    return cos_lib[0][0]

# Search technique from pre-indexing
def search(request):
    results = {}
    with open("./indexing.csv") as f:
        reader = csv.reader(f)

        for row in reader:
            features = [float(x) for x in row[1:]]
            d = compareCosineSimilarity(features, request)
            results[row[0]] = d

        f.close()

    results = sorted([(v, k) for (k, v) in results.items()],reverse=True)
    return results[:10]


# Read path
img_path = './dataset/testing/t1.jpg'

# Using pre-trained VGG16 model to extract feature
model = VGG16(weights='imagenet', include_top=False)

# Preprocess images
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

# Extract feature from image and flatten it into n-dimensional 
vgg16_feature = model.predict(img_data)
features = np.array(vgg16_feature).flatten()

# Looking for the result from DB
results = search(features)

# Display the result ==> Up to researcher 
# The owner is just want to show only the best result
for (score, resultID) in results:
    result = cv2.imread("./dataset/indexing/" + resultID)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    break




