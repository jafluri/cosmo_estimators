from pygsp import filters
import tensorflow as tf

import horovod.tensorflow as hvd

import numpy as np

from shutil import rmtree
import os

# local imports
from .estimators import estimator_1st_order

class BaseModel(object):
    """
    This is a base model, that provides a minimal training step and restore, saving stuff, possibly ditributed...
    """

    def __init__(self, network, input_shape=None, optimizer=None, save_dir=None,
                 restore_point=None, summary_dir=None, init_step=0, is_chief=True):
        """
        Initializes a base model
        :param network: The underlying network of the model (expected to be either a tf.keras.Sequential or a subclass
                        of it)
        :param input_shape: Optional input shape of the network, necessary if one wants to restore the model
        :param optimizer: Optimizer of the model, defaults to Adam
        :param save_dir: Directory where to save the weights and so, can be None
        :param restore_point: Possible restore point, either directory (of which the latest checkpoint will be chosen)
                              or a checkpoint file
        :param summary_dir: Directory to save the summaries
        :param init_step: Initial step, defaults to 0
        :param is_chief: Chief in case of distributed setting
        """

        # get the network
        self.network = network

        # save additional variables
        self.save_dir = save_dir
        self.restore_point = restore_point
        self.summary_dir = summary_dir
        self.input_shape = input_shape
        self.is_chief = is_chief
        self.init_step = init_step

        # set up save dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        # set up the optimizer
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        else:
            self.optimizer = optimizer

        # We build the network
        if self.input_shape is not None:
            self.build_network(input_shape=self.input_shape)
            self.print_summary()

        # restore the weights
        if self.restore_point is not None:
            if input_shape is None:
                print("WARNING: Network weights can't be restored until <build_network> is called! Either call this "
                      "function manually or provide an inpute_shape during model initialization!")
            else:
                print(f"Restoring weights from {self.restore_point}...")
                self.restore_model()

        # set up summary writer
        if self.summary_dir is not None:
            # check if we are distributed
            try:
                _ = hvd.size()
                rank = f"_{hvd.rank()}"
            except ValueError:
                rank = ""

            # make a directory for the writer
            os.makedirs(self.summary_dir + rank, exist_ok=True)
            self.summary_writer = tf.summary.create_file_writer(summary_dir + rank)
        else:
            self.summary_writer = None

        # set the step
        self.train_step = tf.Variable(self.init_step, trainable=False, name="GlobalStep", dtype=tf.int64)
        tf.summary.experimental.set_step(self.train_step)

        # estimator
        self.estimator = None

    def clean_summaries(self, force=False):
        """
        Removes redundant summary directories...
        :param force: force the removal even if the worker is not chief
        """
        if self.is_chief or force:
            try:
                num_workers = hvd.size()
            except ValueError:
                print("Nothing to clean, skipping...")
                num_workers = 1

            for i in range(1, num_workers):
                rmtree(self.summary_dir + f"_{i}")
        else:
            print("Trying to call <clean_summaries> from a worker that is not chief, "
                  "skipping to avoid multiple worker doing the same thing... "
                  "If you are sure that this should happen set force=True when calling this function.")

    def update_step(self):
        """
        increments the train step of the model by 1
        """
        self.train_step.assign(self.train_step + 1)

    def set_step(self, step):
        """
        Sets the current training step of the model to a given value
        :param step: The new step (int)
        """
        self.train_step.assign(step)

    def restore_model(self, restore_point=None):
        """
        Restores the weights of the network given a restore point
        :param restore_point: either a directory that includes checkpoints (of which the most recent will be chosen)
                              or the path to a specific checkpoint to restore from, default to value at init of model
        """

        if restore_point is None and self.restore_point is None:
            raise ValueError("No restore point was provided in the initialization or as argument when this function "
                             "was called.")

        # get the right point
        if restore_point is None:
            restore_point = self.restore_point

        # get the checkpoint
        if os.path.isdir(restore_point):
            checkpoint = tf.train.latest_checkpoint(restore_point)
        else:
            checkpoint = restore_point

        # restore
        self.network.load_weights(checkpoint)

        # check if we need to broadcast
        try:
            num_workers = hvd.size()
            hvd.broadcast_variables(self.network.weights, root_rank=0)
        except ValueError:
            num_workers = 1

        if self.is_chief:
            print(f"Sucessfully resteored weights for {num_workers} workers...")

    def save_model(self, save_dir=None, force=False):
        """
        Saves the weights of the model into a given directory, this function won't do anything if the model is not chief
        :param save_dir: the path where to save the weights, defaults to the value at init of model
        :param force: write the checkpoint even if the model is not chief, this can lead to errors if multiple workers
                      write in the same directory concurrently
        """

        if self.is_chief or force:
            if save_dir is None and self.save_dir is None:
                raise ValueError("No save directory was declared during the init of the model or in this function "
                                 "call.")

            # get and create
            if save_dir is None:
                save_dir = self.save_dir
            os.makedirs(save_dir, exist_ok=True)

            # save
            check_point = os.path.join(save_dir, "checkpoint-%i" % (self.train_step.value()))
            self.network.save_weights(check_point)
        else:
            print("Trying to write a checkpoint with a model that is not chief, skipping... "
                  "If you are sure that this should happen set force=True when calling this function.")

    def build_network(self, input_shape):
        """
        Builds the internal HealpyGCNN with a given input shape
        :param input_shape: input shape of the netork
        """
        self.network.build(input_shape=input_shape)

    def print_summary(self, **kwargs):
        """
        Prints the summary of the internal network
        :param kwargs: passed to HealpyGCNN.summary
        """
        self.network.summary(**kwargs)

    def base_train_step(self, input_tensor, loss_function, input_labels=None, clip_by_value=None, clip_by_norm=None,
                        clip_by_global_norm=None, training=True, num_workers=None, train_indices=None):
        """
        A base train step given a loss funtion and an input tensor it evaluates the network and performs a single
        gradient decent step, if multiple clippings are requested the order will be:
            * by value
            * by norm
            * by global norm
        :param input_tensor: The input of the network
        :param loss_function: The loss function, a callable that takes predictions of the network as input and,
                              if provided, the input_labels
        :param input_labels: Labels of the input_tensor
        :param clip_by_value: Clip the gradients by given 1d array of values into the interval [value[0], value[1]],
                              defaults to no clipping
        :param clip_by_norm: Clip the gradients by norm, defaults to no clipping
        :param clip_by_global_norm: Clip the gradients by global norm, defaults to no clipping
        :param training: whether we are training or not (e.g. matters for batch norm), should be true here
        :param num_workers: how many replicates are working on the same thing, None means no distribution
        :param train_indices: A list of indices, if not None only [trainable_variables[i] for i in train_indices] will
                              be trained
        """
        if train_indices is  None:
            train_variables = self.network.trainable_variables
        else:
            train_variables = [self.network.trainable_variables[i] for i in train_indices]

        with tf.GradientTape() as tape:
            predictions = self.network(input_tensor, training=training)
            if input_labels is None:
                loss_val = loss_function(predictions)
            else:
                loss_val = loss_function(predictions, input_labels)
            if self.summary_writer is not None:
                with self.summary_writer.as_default():
                    tf.summary.scalar("Loss", loss_val)
            # update the step
            self.update_step()

        if num_workers is not None:
            # Horovod: add Horovod Distributed GradientTape.
            tape = hvd.DistributedGradientTape(tape)

        # get the gradients
        gradients = tape.gradient(loss_val, train_variables)

        # clip
        if clip_by_value is not None:
            gradients = [tf.clip_by_value(g, clip_by_value[0], clip_by_value[1]) for g in gradients]
        if clip_by_norm is not None:
            gradients = [tf.clip_by_norm(g, clip_by_norm) for g in gradients]
        # get the global norm
        glob_norm = tf.linalg.global_norm(gradients)
        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("Global_Grad_Norm", glob_norm)
        if clip_by_global_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_by_global_norm, use_norm=glob_norm)

        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, train_variables))

    def broadcast_variables(self):
        """
        boradcasts the variables from the chief to all other workers from the network and optimizer
        """
        hvd.broadcast_variables(self.network.weights, root_rank=0)
        hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

    def setup_1st_order_estimator(self, dset, fidu_param, off_sets, print_params=True, tf_dtype=tf.float32,
                                tikohnov=0.0, layer=None, dset_is_sims=False):
        """
        Sets up a first order estimator from a given dataset that will be evaluated
        :param dset: The dataset that will be evaluated
        :param fidu_param: the fiducial parameter of the estimator
        :param off_sets: the offsets used for the perturbations
        :param print_params: print the calculated params
        :param tf_dtype: the tensorflow datatype to use
        :param tikohnov: Add tikohnov regularization before inverting the jacobian
        :param layer: integer, propagate only up to this layer, can be -1
        :param dset_is_sims: If Ture, dset will be treated as evaluations
        """
        # set the layer
        self.estimator_layer = layer

        # dimension check
        fidu_param = np.atleast_2d(fidu_param)
        n_param = fidu_param.shape[-1]
        n_splits = 2 * n_param + 1

        if dset_is_sims:
            predictions = dset
        else:
            # get the predictions
            predictions = []
            for batch in dset:
                predictions.append(np.split(self.__call__(batch, training=False, layer=self.estimator_layer).numpy(),
                                            indices_or_sections=n_splits, axis=0))
            # concat
            predictions = np.concatenate(predictions, axis=1)

        self.estimator = estimator_1st_order(sims=predictions, fiducial_point=fidu_param, offsets=off_sets,
                                             print_params=print_params, tf_dtype=tf_dtype, tikohnov=tikohnov)

    def estimate(self, input_tensor):
        """
        Calculates the first order estimates of the underlying model parameter given a network input
        :param input_tensor: The input to feed in the network
        :return: The parameter estimates
        """

        if self.estimator is None:
            raise ValueError("First order estimator not set! Call <setup_1st_order_estimator> first!")

        preds = self.__call__(input_tensor, training=False, layer=self.estimator_layer)
        return self.estimator(preds)

    def __call__(self, input_tensor, training=False, numpy=False, layer=None, *args, **kwargs):
        """
        Calls the underlying network
        :param input_tensor: the tensor (or array) to call on
        :param training: whether we are training or evaluating (e.g. necessary gor batch norm)
        :param args: additional arguments passed to the network
        :param kwargs: additional keyword arguments passed to the network
        :param numpy: return a numpy array instead of a tensor
        :param layer: integer, propagate only up to this layer, can be -1
        :return: either a tensor or an array depending on param numpy
        """
        if layer is None:
            preds = self.network(input_tensor, training=training, *args, **kwargs)
        else:
            preds = input_tensor
            for layer in self.network.layers[:layer]:
                preds = layer(preds)

        if numpy:
            return preds.numpy()
        else:
            return preds
