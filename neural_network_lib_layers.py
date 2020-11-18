import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

##tf.device("gpu:0")
import keras
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras import regularizers
from keras.models import Model
from keras import metrics


class PolicyValueNetwork:

    def __init__(self, l2_const, starting_network_file=None, train_supervised=False):

        self.l2_const = l2_const

        # If a model is already provided, load it from the file.
        if starting_network_file:
            self.model = keras.models.load_model(starting_network_file)
            if train_supervised:
                self.model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                                   loss={"value": keras.losses.mean_squared_error, "policy": keras.losses.categorical_crossentropy},
                                   loss_weights=[0.01, 1.0],
                                   metrics=[metrics.mae, metrics.categorical_accuracy])
            return

        # Otherwise, build the model. It will have 19 layers and process a board of size 13x13.
        inputs = Input(shape=(13, 13, 19))

        # Define the input convolutional block that will be used by the network.
        conv_block = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same",
                            kernel_initializer='random_normal',
                            bias_initializer='random_normal',
                            kernel_regularizer=regularizers.l2(l2_const),
                            bias_regularizer=regularizers.l2(l2_const))(inputs)
        conv_block = BatchNormalization()(conv_block)
        conv_block = Activation("relu")(conv_block)

        # The policy and value blocks will connect to a stack of 11 residual connection blocks.
        res = conv_block
        for i in range(11):
            res = self.create_res_block(res)

        # The policy block will have an output size of 170 (169 board positions, and one option to pass).
        policy = Conv2D(filters=2,
                        kernel_size=(1, 1),
                        strides=1,
                        padding="same",
                        kernel_initializer='random_normal',
                        bias_initializer='random_normal',
                        kernel_regularizer=regularizers.l2(l2_const),
                        bias_regularizer=regularizers.l2(l2_const))(res)
        policy = BatchNormalization()(policy)
        policy = Activation("relu")(policy)
        policy = Reshape((13*13*2,))(policy)
        policy = Dense(170, activation="softmax", name="policy")(policy)

        # The value block will have an output size of 1, indicating the outcome of the game (win or loss).
        value = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=1,
                       padding="same",
                       kernel_initializer='random_normal',
                       bias_initializer='random_normal',
                       kernel_regularizer=regularizers.l2(l2_const),
                       bias_regularizer=regularizers.l2(l2_const))(res)
        value = BatchNormalization()(value)
        value = Activation("relu")(value)
        value = Reshape((13*13,))(value)
        value = Dense(64, activation="relu")(value)
        value = Dense(1, activation="tanh", name="value")(value)

        # Construct the model by combining the inputs and the value / policy outputs.
        self.model = Model(inputs=inputs, outputs=[value, policy])

        # Compile the model. If training with supervised learning, give greater weight to the policy network.
        if train_supervised:
            self.model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                               loss={"value": keras.losses.mean_squared_error,
                                     "policy": keras.losses.categorical_crossentropy},
                               loss_weights=[0.01, 1.0],
                               metrics=[metrics.mae, metrics.categorical_accuracy])
        else:
            self.model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                               loss={"value": keras.losses.mean_squared_error,
                                     "policy": keras.losses.categorical_crossentropy},
                               loss_weights=[1.0, 1.0],
                               metrics=[metrics.mae, metrics.categorical_accuracy])


        print(self.model.summary())

        #self.model._make_predict_function()

        # Save the model so we can load it in a different thread!!
        #self.save_model_to_file("young_saigon.h5")

    def load_latest_model(self):
        self.model = keras.models.load_model("young_saigon.h5")

    def create_res_block(self, input_layer):
        res_conn = input_layer
        res_block = Conv2D(filters=64,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           kernel_initializer='random_normal',
                           bias_initializer='random_normal',
                           kernel_regularizer=regularizers.l2(self.l2_const),
                           bias_regularizer=regularizers.l2(self.l2_const))(input_layer)
        res_block = BatchNormalization()(res_block)
        res_block = Activation("relu")(res_block)
        res_block = Conv2D(filters=64,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           kernel_initializer='random_normal',
                           bias_initializer='random_normal',
                           kernel_regularizer=regularizers.l2(self.l2_const),
                           bias_regularizer=regularizers.l2(self.l2_const))(res_block)
        res_block = BatchNormalization()(res_block)
        res_block = keras.layers.add([res_conn, res_block])
        res_block = Activation("relu")(res_block)
        return res_block

    def predict_given_state(self, model_inputs, batch_size=1):
        pred_value, prior_probs = self.model.predict(model_inputs, batch_size=batch_size)
        return prior_probs, pred_value

    def print_model(self):
        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png')
        keras.utils.print_summary(self.model)

    def save_model_to_file(self, model_filename):
        self.model.save(model_filename)

    def train(self, training_data_input, training_data_gt_value, training_data_gt_policy, save_file):
        """
        This function creates a copy of the model, trains it, and then replaces the model when training is done.
        :param training_data: a numpy array of
        """
        # Load a copy of the model for training.
        training_model = keras.models.load_model(save_file)
        training_model.fit(x=training_data_input,
                           y={"value": training_data_gt_value, "policy": training_data_gt_policy},
                           batch_size=32,
                           epochs=3,
                           verbose=2)

        # TODO: Watch out for race conditions!!! Obtain a lock to update self.model.
        # Save the newly trained version of this model to the young_saigon.h5 file.
        training_model.save(save_file)
        print("Done training!")

    def train_supervised(self, training_data_input, training_data_gt_value, training_data_gt_policy, save_file):
        """
        This function creates a copy of the model, trains it, and then replaces the model when training is done.
        :param training_data: a numpy array of
        """
        print(training_data_input.shape)
        print(training_data_gt_value.shape)
        print(training_data_gt_policy.shape)
        # Load a copy of the model for training.
        self.model.fit(x=training_data_input,
                       y={"value": training_data_gt_value, "policy": training_data_gt_policy},
                       epochs=1, verbose=2,
                       batch_size=8)

        # TODO: Watch out for race conditions!!! Obtain a lock to update self.model.
        # Save the newly trained version of this model to the young_saigon.h5 file.
        # self.model.save(save_file)
        print("Done training!")

    def save_checkpoint(ckpt_file):
        self.model.save(ckpt_file)


