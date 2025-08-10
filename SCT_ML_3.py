from matplotlib.pyplot import imshow, show
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Assuming model and Categories are already defined
# Example:
# model = trained_model
# Categories = ['Cat', 'Dog']

path = 'dataset/test_set/dogs/dog.4001.jpg'

# Read and display image
img = imread(path)
plt.imshow(img)
plt.axis('off')
plt.show()

# Resize image and flatten
img_resize = resize(img, (150, 150, 3))
l = [img_resize.flatten()]

# Predict probabilities
probability = model.predict_proba(l)

# Display each category's probability
for ind, val in enumerate(Categories):
    print(f'{val} = {probability[0][ind] * 100:.2f}%')

# Print predicted category
predicted_label = model.predict(l)[0]
print("The predicted image is : " + Categories[predicted_label])
