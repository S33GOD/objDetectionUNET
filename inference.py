import matplotlib.pyplot as plt
import random
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import cv2

IMAGE_DIR = "images_train_dataset/"
MASK_DIR = "images_masks_dataset/"
image_names = glob.glob(IMAGE_DIR + "*.jpg")
image_names.sort()
images = [cv2.resize(cv2.imread(img,0), (128,128), interpolation=cv2.INTER_AREA) for img in image_names]
images_dataset = np.array(images)
images_dataset = np.expand_dims(images_dataset, axis=3)

mask_names = glob.glob(MASK_DIR + "*.tiff")
mask_names.sort()
masks = [cv2.resize((cv2.imread(mask, 0)), (128,128), interpolation=cv2.INTER_AREA) for mask in mask_names]
masks_dataset = np.array(masks)
masks_dataset = np.expand_dims(masks_dataset, axis=3)

X_train, X_test, y_train, y_test = train_test_split(images_dataset, masks_dataset, test_size= 0.9, random_state=42)
model = load_model("unet_500ep.hdf5", custom_objects={"dice_coef" : "dice_coef"})

test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
print(test_img_input.shape)
prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.8).astype(np.uint8)
prediction = prediction/255.
print(prediction.shape)

plt.figure(figsize=(16,8))
plt.subplot(231)
plt.title("Image")
plt.imshow(test_img[:,:,0], cmap="gray")
plt.subplot(232)
plt.title("Label")
plt.imshow(ground_truth[:,:,0], cmap="gray")
plt.subplot(233)
plt.title("Prediction")
plt.imshow(prediction, cmap="gray")
plt.show()