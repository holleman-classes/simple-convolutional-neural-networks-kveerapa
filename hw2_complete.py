### Add lines to import modules as needed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Add, Dropout, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from google.colab import files
from PIL import Image
## 

def build_model1():
  model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),

        # Four more pairs of Conv2D+Batchnorm
        tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=4,padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  return model

def build_model2():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False),
        tf.keras.layers.Conv2D(64, kernel_size=1, strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False),
        tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False),
        tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False),
        tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False),
        tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False),
        tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),



        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model

def build_model3():
    inputs = layers.Input(shape=(32, 32, 3))

    y = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    y = layers.Conv2D(128, (1, 1), strides=(5, 5), padding='same', activation='relu')(y)
    y = layers.Add()([x, y])
    y = layers.Dropout(0.5)(y)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    y = layers.Add()([x, y])
    y = layers.Dropout(0.5)(y)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(y)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    y = layers.Add()([x, y])
    y = layers.Dropout(0.5)(y)

    x = layers.MaxPooling2D(pool_size=(4, 4))(y)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(10)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model
def build_model50k():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (2, 2), strides=(2, 2), padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.SeparableConv2D(64, kernel_size=(3, 3), activation="relu", strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, kernel_size=(2, 2), activation="relu", strides=(2, 2), padding='same'),
        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.SeparableConv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':



  ########################################
  ## Add code here to Load the CIFAR10 data set
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    val_split = 0.2  # 20% for validation
    num_train_examples = len(train_images)
    num_val_examples = int(val_split * num_train_examples)

    val_images = train_images[:num_val_examples]
    val_labels = train_labels[:num_val_examples]
    train_images = train_images[num_val_examples:]
    train_labels = train_labels[num_val_examples:]

  # Normalize the pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
# Convert the labels to one-hot encoded vectors
    num_classes = 10
    train_labels = np.eye(num_classes)[train_labels.squeeze()]
    val_labels = np.eye(num_classes)[val_labels.squeeze()]
    test_labels = np.eye(num_classes)[test_labels.squeeze()]

  ############################################################################# MODEL 1 #################################################
  ## Build and train model 1
    model1 = build_model1()
    model1.summary()
    # compile and train model 1.
    # Train the model for 50 epochs
    history_model1 = model1.fit(train_images, train_labels, epochs=50, batch_size=128, validation_data=(val_images, val_labels))
    # Plot the training and validation accuracy
    plt.plot(history_model1.history['accuracy'])
    plt.plot(history_model1.history['val_accuracy'])
    plt.title('Model 1 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot the training and validation loss
    plt.plot(history_model1.history['loss'])
    plt.plot(history_model1.history['val_loss'])
    plt.title('Model 1 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy:', test_acc)
    #Saving the trained model
    model1.save('model1.h5')
 


    img = Image.open("dog.jpg") # Loading Image
    img = img.resize((32, 32)) # Resizing
    x = np.array(img) # Convert the image to a numpy array
    x = np.expand_dims(x, axis=0)
    model = keras.models.load_model("model1.h5")
    # Make a prediction on the image
    pred = model.predict(x)
    # Get the predicted class label
    class_index= np.argmax(pred)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = class_names[class_index]
    print(class_name)



  ############################################################################ MODEL 2 #####################################################

  ## Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()
    model2.summary()


    # Train the model for 50 epochs
    history_model2 = model2.fit(train_images, train_labels, epochs=50, batch_size=128, validation_data=(val_images, val_labels))

    # Plot the training and validation accuracy
    plt.plot(history_model2.history['accuracy'])
    plt.plot(history_model2.history['val_accuracy'])
    plt.title('Model 2 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot the training and validation loss
    plt.plot(history_model2.history['loss'])
    plt.plot(history_model2.history['val_loss'])
    plt.title('Model 2 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_acc = model2.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy:', test_acc)
    #Saving the trained model
    model2.save('model2.h5')


    img = Image.open("dog.jpg") # Loading Image
    img = img.resize((32, 32)) # Resizing
    x = np.array(img) # Convert the image to a numpy array
    x = np.expand_dims(x, axis=0)
    model = keras.models.load_model("model2.h5")
    # Make a prediction on the image
    pred = model.predict(x)
    # Get the predicted class label
    class_index= np.argmax(pred)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = class_names[class_index]
    print(class_name)

    ############################################################################# MODEL 3 ###################################################
  
  ### Repeat for model 3 and your best sub-50k params model
    model3=build_model3()
    model3.summary()
    # Train the model for 50 epochs
    history_model3 = model3.fit(train_images, train_labels, epochs=50, batch_size=128, validation_data=(val_images, val_labels))

    # Plot the training and validation accuracy
    plt.plot(history_model3.history['accuracy'])
    plt.plot(history_model3.history['val_accuracy'])
    plt.title('Model 3 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot the training and validation loss
    plt.plot(history_model3.history['loss'])
    plt.plot(history_model3.history['val_loss'])
    plt.title('Model 3 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_acc = model3.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy:', test_acc)
    #Saving the trained model
    model3.save('model3.h5')
    #uploaded = files.upload()
    img = Image.open("dog.jpg") # Loading Image
    img = img.resize((32, 32)) # Resizing
    x = np.array(img) # Convert the image to a numpy array
    x = np.expand_dims(x, axis=0)
    model = keras.models.load_model("model3.h5")
    # Make a prediction on the image
    pred = model.predict(x)
    # Get the predicted class label
    class_index= np.argmax(pred)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = class_names[class_index]
    print(class_name)


   ## Build and train model 3
    model50k = build_model50k()
    model50k.summary()

    # Train the model for 50 epochs
    history_model50k = model50k.fit(train_images, train_labels, epochs=50, batch_size=128, validation_data=(val_images, val_labels))

    # Plot the training and validation accuracy
    plt.plot(history_model50k.history['accuracy'])
    plt.plot(history_model50k.history['val_accuracy'])
    plt.title('Model 50K accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot the training and validation loss
    plt.plot(history_model50k.history['loss'])
    plt.plot(history_model50k.history['val_loss'])
    plt.title('Model 50K loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_acc = model50k.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy:', test_acc)
    #Saving the trained model
    model3.save('best_model.h5')


    #uploaded = files.upload()
    img = Image.open("dog.jpg") # Loading Image
    img = img.resize((32, 32)) # Resizing
    x = np.array(img) # Convert the image to a numpy array
    x = np.expand_dims(x, axis=0)
    model = keras.models.load_model("best_model.h5")
    # Make a prediction on the image
    pred = model.predict(x)
    # Get the predicted class label
    class_index= np.argmax(pred)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = class_names[class_index]
    print(class_name)
  
  

