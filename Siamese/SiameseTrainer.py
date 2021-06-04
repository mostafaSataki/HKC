import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from  ..FileUtility import *
from  .DistanceLayer import  *
from .SiameseModel import  *
from tensorflow.keras import optimizers

class SiameseTrainer:
    def __init__(self):
        self.target_shape_ = (200,200)
        self.batch_size_ = 32

    # make pairs
    def make_pairs(self, x, y):
        y_arr = np.array(y)
        labels_name = list( set(y))
        num_classes = len(labels_name)
        indexs_name = list(range(num_classes))
        labels_dict = {}
        for i ,label_name in enumerate( labels_name):
            labels_dict[label_name] = indexs_name[i]

        digit_indices = [np.where(y_arr == i,)[0] for i in  labels_name]
        indexs = []
        for i in y:
           indexs.append(labels_dict[i])


        anchors = []
        positives = []
        negatives = []

        pairs = []
        labels = []

        for idx1 in range(len(x)):
            # add a matching example
            x1 = x[idx1]
            label1 = y[idx1]

            idx2 = random.choice(digit_indices[labels_dict[label1]])
            x2 = x[idx2]

            label2 = random.randint(0, num_classes - 1)
            while labels_name[label2] == label1:
                label2 = random.randint(0, num_classes - 1)

            # idx2 = random.choice(digit_indices[labels_dict[labels_name[label2]]])
            idx2 = random.choice(digit_indices[label2])
            x3 = x[idx2]

            anchors.append(x1)
            positives.append(x2)
            negatives.append(x3)

        c = list(zip(anchors, positives,negatives))

        random.shuffle(c)

        anchors, positives,negatives  = [list(tuple) for tuple in zip(*c)]
        return anchors, positives,negatives

    def preprocess_image(self,filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image,self.target_shape_)
        return image

    def preprocess_triplets(self,anchor, positive, negative):
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """

        return (
            self.preprocess_image(anchor),
            self.preprocess_image(positive),
            self.preprocess_image(negative),
        )

    def make_dataset(self,anchors, positives,negatives,per = 0.8):


        image_count = len(anchors)

        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchors)

        positive_dataset = tf.data.Dataset.from_tensor_slices(positives)
        negative_dataset = tf.data.Dataset.from_tensor_slices(negatives)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))

        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.map(self.preprocess_triplets)

        # Let's now split our dataset in train and validation.
        train_dataset = dataset.take(round(image_count * per))
        val_dataset = dataset.skip(round(image_count * per))

        train_dataset = train_dataset.batch(self.batch_size_, drop_remainder=False)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = val_dataset.batch(self.batch_size_, drop_remainder=False)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return  train_dataset, val_dataset
    def getEmbeddingModel(self):
        base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=self.target_shape_ + (3,), include_top=False
        )

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        embedding = Model(base_cnn.input, output, name="Embedding")

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        return  embedding

    def getInputLayers(self):
        anchor_input = layers.Input(name="anchor", shape=self.target_shape_ + (3,))
        positive_input = layers.Input(name="positive", shape=self.target_shape_ + (3,))
        negative_input = layers.Input(name="negative", shape=self.target_shape_ + (3,))

        return  anchor_input,positive_input,negative_input

    def visualize(self, anchor, positive, negative):
        """Visualize a few triplets from the supplied batches."""

        def show(ax, image):
            ax.imshow(image)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # fig = plt.figure(figsize=(9, 9))
        #
        # axs = fig.subplots(3, 3)
        # for i in range(3):
        #     cv2.imshow("anchor"+str(i),anchor[i])
        #     cv2.imshow("positive"+str(i), positive[i])
        #     cv2.imshow("negative"+str(i), negative[i])
        #     cv2.waitKey(0)
        plt.subplot(1, 2, 1)
        plt.imshow(anchor[0])
        plt.subplot(1, 2, 2)
        plt.imshow(positive[0])





    def _testProc(self,sample):
        anchor, positive, negative = sample
        anchor_embedding, positive_embedding, negative_embedding = (
            self.embedding(resnet.preprocess_input(anchor)),
            self.embedding(resnet.preprocess_input(positive)),
            self.embedding(resnet.preprocess_input(negative)),
        )


        cosine_similarity = metrics.CosineSimilarity()

        positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
        print("Positive similarity:", positive_similarity.numpy())

        negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
        print("Negative similarity", negative_similarity.numpy())

    def train(self,src_path):
        filenames, labels = FileUtility.loadFilenamesLabels(src_path)

        anchors, positives, negatives = self.make_pairs(filenames, labels)
        train_dataset, val_dataset = self.make_dataset(anchors, positives, negatives)

        sample = next(iter(train_dataset))
        # self.visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])


        self.embedding = self.getEmbeddingModel()

        anchor_input, positive_input, negative_input = self.getInputLayers()

        distances = DistanceLayer()(
            self.embedding(resnet.preprocess_input(anchor_input)),
            self.embedding(resnet.preprocess_input(positive_input)),
            self.embedding(resnet.preprocess_input(negative_input)),
        )

        siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

        siamese_model = SiameseModel(siamese_network)
        siamese_model.compile(optimizer=optimizers.Adam(0.0001))
        siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

        sample = next(iter(train_dataset))
        self.visualize(*sample)

        self._testProc(sample)

