import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pickle
import ipdb
import numpy as np


# load dataset function (dog-cat image 32*32)
def open_catdog_dataset():

    data_path = ['./data/cat_dog_train_32.pkl','./data/cat_dog_test_32.pkl']
    
    with open(data_path[0], 'rb') as f:
        trainset = pickle.load(f)
    with open(data_path[1], 'rb') as f:
        testset = pickle.load(f)

    return trainset, testset

def shuffle_dataset(data_list, label_list):
    # shuffle index
    index_list = np.arange(len(label_list))
    np.random.shuffle(index_list)

    # shuffle list
    shuffled_data = data_list[index_list]
    shuffled_label = label_list[index_list]

    return shuffled_data, shuffled_label


# load dataset
(train_images, train_labels), (test_images, test_labels) = open_catdog_dataset()

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# shuffle dataset
train_images, train_labels = shuffle_dataset(train_images, train_labels)

# data check (cat, dog) 
class_names = ['cat', 'dog']

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

ipdb.set_trace()

# augmentation layer
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

### make neural network model
model = models.Sequential()
# model.add(data_augmentation)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.0))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.0))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

### add dense layer (classification layer)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()


### training setting part
checkpoint_path = './checkpoint/test.ckpt'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            monitor='val_accuracy',
                                            save_weights_only=True,
                                            mode='max',
                                            save_best_only=True,
                                            verbose=1)

### compile => optimizer, loss function, mectric, ...
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

### model training part
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels),
                    callbacks=[checkpoint])

### model evaluation part
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
