from IPython.display import Image, display
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.applications import vgg16

from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, Sequential
from keras import optimizers
import matplotlib.pyplot as plt


def define_cnn_model():
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 300

    model=Sequential()

    #model.add(Lambda(standardize,input_shape=(28,28,1)))    
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
        
    model.add(MaxPooling2D(pool_size=(2,2)))
        
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="relu"))
        
    model.add(Dense(10,activation="softmax"))
        
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def define_transfer_model():
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 300
    model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), pooling="max")

    for layer in model.layers[:-5]:
            layer.trainable = False
            
    for layer in model.layers:
        print(layer, layer.trainable)


    transfer_model = Sequential()
    for layer in model.layers:
        transfer_model.add(layer)
    transfer_model.add(Dense(512, activation="relu"))  
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(10, activation="softmax")) 

    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)

    transfer_model.compile(loss="categorical_crossentropy",
                          optimizer=adam,
                          metrics=["accuracy"])

    return transfer_model


def summarize_diagnostics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.show()
    # filename = sys.argv[0].split('/')[-1]
    plt.savefig('graphs.png')
    plt.close()
    


TRAIN_DIR = "../data/training/training/"
VAL_DIR = "../data/validation/validation/"
# print(os.listdir(TRAIN_DIR))


IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
BATCH_SIZE = 16
train_samples = 1097
val_samples = 272

train_datagen = ImageDataGenerator(rescale=1./255,      # We need to normalize the data
                                    rotation_range=40,      # The rest of params will generate us artificial data by manipulating the image
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True, 
                                    fill_mode='nearest'
                                  )

val_datagen = ImageDataGenerator(rescale=1./255, # We need to normalize the data
                                  )

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                                                    batch_size = BATCH_SIZE, 
                                                    shuffle=True, # By shuffling the images we add some randomness and prevent overfitting
                                                    class_mode="categorical")

val_gen = val_datagen.flow_from_directory(VAL_DIR, 
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                                                    batch_size = BATCH_SIZE, 
                                                    shuffle=True,
                                                    class_mode="categorical")



model = define_transfer_model()
# model = define_cnn_model()



history = model.fit_generator(train_gen, steps_per_epoch=train_samples // BATCH_SIZE,
                                            epochs=40,
                                            validation_data=val_gen,
                                            validation_steps=val_samples // BATCH_SIZE)


model.save('model.h5')
# model = load_model('model.h5')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

_, acc = model.evaluate_generator(val_gen, steps=val_samples // BATCH_SIZE, verbose=0)
print('Testing Accuracy: %.3f' % (acc * 100.0))

_, train_acc = model.evaluate_generator(train_gen, steps=train_samples // BATCH_SIZE, verbose=0)
print('Training Accuracy: %.3f' % (train_acc * 100.0))


summarize_diagnostics(history)

# fig = plt.figure(figsize=(10, 10)) # Set Figure

# y_pred = model.predict(X_test) 
# Y_pred = np.argmax(y_pred, 1) 
# Y_test = np.argmax(y_test, 1) 

# mat = confusion_matrix(Y_test, Y_pred) # Confusion matrix

# # Plot Confusion matrix
# sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
# plt.xlabel('Predicted Values')
# plt.ylabel('True Values');
# # plt.show();
# plt.savefig('confusion_matrix.png')
# plt.close()

# X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

# fig, axis = plt.subplots(4, 4, figsize=(12, 14))
# for i, ax in enumerate(axis.flat):
#     ax.imshow(X_test__[i], cmap='binary')
#     ax.set(title = f"Real Number is {y_test[i].argmax()}\nPredicted Number is {y_pred[i].argmax()}");

# plt.savefig('results.png')
# plt.close()