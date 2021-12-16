from sklearn.model_selection import train_test_split
import glob
import cv2
import numpy as np
import keras.backend as K
from keras.layers import LayerNormalization
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Input, BatchNormalization, Dropout, Lambda
from keras.optimizers import adam_v2
from keras.layers import Activation, MaxPool2D, Concatenate

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

images_dataset = images_dataset/255.

X_train, X_test, y_train, y_test = train_test_split(images_dataset, masks_dataset, test_size= 0.9, random_state=42)



def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2,2))(x)
    return x,p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    activation = "sigmoid"

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model

IMG_HEIGHT = images_dataset.shape[1]
IMG_WIDTH = images_dataset.shape[2]
IMG_CHANNELS = images_dataset.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union+smooth), axis=0)
    return dice

model = build_unet(input_shape, n_classes= 1)
model.compile(optimizer=adam_v2.Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=[dice_coef])
model.summary()

history = model.fit(X_train, y_train,
                    batch_size = 16,
                    verbose = 1,
                    epochs = 500,
                    validation_data = (X_test, y_test),
                    shuffle = False)
model.save("unet_500ep.hdf5")