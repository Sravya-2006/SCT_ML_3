from matplotlib.pyplot import imshow, show
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import numpy as np
path = 'dataset/test_set/dogs/dog.4001.jpg'
img = imread(path)
plt.imshow(img)
plt.axis('off')
plt.show()
img_resize = resize(img, (150, 150, 3))
l = [img_resize.flatten()]
probability = model.predict_proba(l)
for ind, val in enumerate(Categories):
    print(f'{val} = {probability[0][ind] * 100:.2f}%')
predicted_label = model.predict(l)[0]
print("The predicted image is : " + Categories[predicted_label])

