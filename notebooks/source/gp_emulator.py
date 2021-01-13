import numpy as np

import GPyOpt

import gpflow

from gpflow.optimizers import NaturalGradient
from gpflow.utilities import print_summary

import tensorflow as tf
import tensorflow_probability as tfp

from time import time

from tqdm import tqdm

import os

class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    """
    Likelihood for varying noise amplitude in the data
    """
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        print(F.shape)
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu = Fmu[:,0]
        Fvar = Fvar[:,0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError


class VGP_Emu():
    # default types
    default_np_float = gpflow.default_float()
    if gpflow.default_float() is np.float32:
        default_tf_float = tf.float32
    else:
        default_tf_float = tf.float64
    default_np_int = gpflow.default_int()
    if gpflow.default_int() is np.int32:
        default_tf_int = tf.int32
    else:
        default_tf_int = tf.int64

    def __init__(self, objective=None, space=None, N_init=20, X_init=None, Y_init=None, normalize_Y=True, mean_only=False,
                 alpha=0.01, kern="matern52", num_restarts=10, verbosity=0, max_opt_iter=1000, full_restart=False,
                 ARD=False, learning_rate=1e-4, parameter_noise_scale=0.1, minimum_variance=1e-3):
        """
        An class that fits a Gaussian process to a given objective function
        :param objective: function used for the fitting (needs to estimate the noise as well!)
        :param space: a GPy space for the prior
        :param N_init: number of initial points
        :param X_init: initial points in space (if set N_init is ignored)
        :param Y_init: optionial with set with X_init the objective is not called on X_init
        :param normalize_Y: normalize the Y coordinate to have zero mean and unit variance (standard)
        :param mean_only: normalize Y only such that it has zero mean but leave std as is
        :param alpha: alpha value in the acquisition function of Raul's paper
        :param kern: kernel type, currently only Matern52 or Exponential
        :param num_restarts: number of restarts for each GP optimization
        :param verbosity: 0 -> print minimal output, higher value = more output
        :param max_opt_iter: maximum iteration for a single GP optimization
        :param full_restart: ignore the current kernel values for the next optimization
        :param ARD: Auto Relevance Determination, use a lengthscale in the kernel for each dimension of the problem
        :param learning_rate: learning rate for the Adam optimizer used for the optimization
        :param parameter_noise_scale: noise std that is added to the parameter for optimization
        :param minimum_variance: minimum of the allowed variance estimate, choosing this too small leads to numerical
        instabilities.
        """

        # some sanity checks
        if (objective is None or space is None) and (X_init is None and Y_init is None):
            raise ValueError("If there is no initial dataset, one has to provide an objective function and a space!")

        self.objective = objective
        self.verbosity = verbosity
        # how to start
        if X_init is None:
            initial_design = GPyOpt.experiment_design.initial_design('latin', space, N_init)
            initial_Y = objective(initial_design)
        elif Y_init is None:
            initial_design = X_init
            initial_Y = objective(initial_design)
        else:
            initial_design = X_init
            initial_Y = Y_init

        # we need to split off the variance estimates
        initial_Y, initial_var = np.split(initial_Y, axis=1, indices_or_sections=2)

        # tfp prior
        self.space = space
        if self.space is not None:
            a_min = np.asarray(self.space.get_bounds(), dtype=self.default_np_float).T[0]
            a_max = np.asarray(self.space.get_bounds(), dtype=self.default_np_float).T[1]
            self.tfp_prior = tfp.distributions.Uniform(low=a_min, high=a_max)

        # normalize
        if normalize_Y:
            self.Y_mean = np.mean(initial_Y)
            if mean_only:
                self.Y_std = 1.0
            else:
                self.Y_std = np.std(initial_Y)
            self.Y_all = (initial_Y - self.Y_mean) / self.Y_std
        else:
            self.Y_mean = 0.0
            self.Y_std = 1.0
            self.Y_all = initial_Y
        self.normalize_Y = normalize_Y
        self.mean_only = mean_only

        # now we need to take care of the variance estimates
        self.var_estimates = initial_var / self.Y_std ** 2

        # normalization
        self.params, self.rot_mat, self.rot_mean, self.rot_std = self.normalize_params(initial_design)

        # kernel
        self.dims = int(X_init.shape[-1])
        self.kern_type = kern
        if ARD:
            lengthscales = [1.0 for _ in range(self.dims)]
        else:
            lengthscales = 1.0
        if kern == "matern52":
            self.kern = gpflow.kernels.Matern52(lengthscales=lengthscales)
        elif kern == "exponential":
            self.kern = gpflow.kernels.Exponential(lengthscales=lengthscales)
        else:
            raise IOError("Unkown kernel")
        self.lengthscale_shape = self.kern.lengthscales.shape

        if num_restarts is None or num_restarts < 1:
            print("Number of restarts is set to 1!")
            self.num_restarts = self.default_np_int(1)
        else:
            self.num_restarts = self.default_np_int(num_restarts)

        # get the likelihood
        self.likelihood = HeteroskedasticGaussian()

        # model (if you get a matrix inversion error here increase number of initial params)
        self.minimum_variance = minimum_variance
        data = np.concatenate([self.Y_all, np.maximum(self.var_estimates, self.minimum_variance)], axis=1)
        self.model = gpflow.models.VGP((self.params.astype(self.default_np_float),
                                        data.astype(self.default_np_float)),
                                       kernel=self.kern, likelihood=self.likelihood, num_latent_gps=1)

        # We turn off training for q as it is trained with natgrad
        gpflow.utilities.set_trainable(self.model.q_mu, False)
        gpflow.utilities.set_trainable(self.model.q_sqrt, False)

        # summary
        print_summary(self.model)

        # save params
        self.learning_rate = learning_rate
        self.parameter_noise_scale = parameter_noise_scale
        self.max_opt_iter = max_opt_iter
        self.full_restart = full_restart
        self.optimize_model()

        # for acquisition
        self.current_transform = lambda x: self.transform_params(x, self.rot_mat, self.rot_mean, self.rot_std)
        self.alpha = alpha

    def acquisition_function(self, x):
        """
        Raul's acquisition function
        """
        if self.current_transform is not None:
            x = self.current_transform(x)

        mean, var = self.model.predict_f(x)

        if self.normalize_Y:
            mean = mean * self.Y_std + self.Y_mean
            var *= self.Y_std ** 2

        return -(tf.exp(mean) + self.alpha * (tf.exp(var) - 1.0) * tf.exp(2 * mean + var))

    def train_model(self, n_iters):
        """
        Optimizes the model for n_iters
        :param n_iters: number of iterations for optimization
        :return: a list of losses of each step with length n_iters
        """

        @tf.function
        def objective_closure():
            return self.model.training_loss()

        natgrad = NaturalGradient(gamma=1.0)
        adam = tf.optimizers.Adam(self.learning_rate)

        print("Training the VGP model params...", flush=True)
        losses = []
        with tqdm(range(n_iters), total=n_iters) as pbar:
            for _ in pbar:
                natgrad.minimize(objective_closure, [(self.model.q_mu, self.model.q_sqrt)])
                adam.minimize(objective_closure, self.model.trainable_variables)
                loss = objective_closure().numpy()
                losses.append(loss)
                pbar.set_postfix(loss_val=loss, refresh=False)

        return losses

    def _readout_params(self):
        """
        Reads out the params of the model and returns them as tupel of arrays
        :return: tupel of params
        """

        params = (self.model.kernel.variance.numpy(), self.model.kernel.lengthscales.numpy(), self.model.q_mu.numpy(),
                  self.model.q_sqrt.numpy())

        return params

    def _set_params(self, params):
        """
        Sets the model params to a given tupel of array
        :param params: params (tupel of arrays)
        """
        self.model.kernel.variance.assign(params[0])
        self.model.kernel.lengthscales.assign(params[1])
        self.model.q_mu.assign(params[2])
        self.model.q_sqrt.assign(params[3])

    def optimize_model(self, scale=1.0):
        """
        Optimizes the model for a given number of restarts and chooses the best result
        :param scale: std of the normal distribution used to draw new params
        """

        func_vals = []
        model_params = []

        # read out the original params
        original_params = self._readout_params()

        for i in range(self.num_restarts):

            # we need to create a new optimizer since Adam has params itself
            self.opt = tf.optimizers.Adam(self.learning_rate)

            try:
                # assign new staring vals
                if self.full_restart:
                    # This is used in GPy opt if no prior is specified (see model.randomize() defined in paramz pack)
                    self.model.kernel.variance.assign(
                        tf.maximum(tf.random.normal(shape=(), dtype=self.default_tf_float, mean=scale,
                                                    stddev=self.parameter_noise_scale),
                                   tf.constant(0.1, dtype=self.default_tf_float)))
                    self.model.kernel.lengthscales.assign(
                        tf.maximum(tf.random.normal(shape=self.lengthscale_shape,
                                                    dtype=self.default_tf_float, mean=scale,
                                                    stddev=self.parameter_noise_scale),
                                   tf.constant(0.1, dtype=self.default_tf_float)))
                    self.model.q_mu.assign(tf.zeros_like(self.model.q_mu))
                    self.model.q_sqrt.assign(tf.eye(len(original_params[2]), batch_shape=[1],
                                                    dtype=self.default_tf_float))
                else:
                    self.model.kernel.variance.assign(tf.maximum(original_params[0] +
                                                                 tf.random.normal(shape=(),
                                                                                  dtype=self.default_tf_float,
                                                                                  stddev=self.parameter_noise_scale),
                                                                 tf.constant(0.1, dtype=self.default_tf_float, )))
                    self.model.kernel.lengthscales.assign(tf.maximum(original_params[1] +
                                                                     tf.random.normal(shape=self.lengthscale_shape,
                                                                                      dtype=self.default_tf_float,
                                                                                      stddev=self.parameter_noise_scale),
                                                                     tf.constant(0.1, dtype=self.default_tf_float)))
                    self.model.q_mu.assign(original_params[2])
                    self.model.q_sqrt.assign(original_params[3])

                # now we optimize
                losses = self.train_model(self.max_opt_iter)

                # we append the final loss value
                func_vals.append(losses[-1])
                model_params.append(self._readout_params())
                if self.verbosity > 0:
                    print("Optimization {}: achieved {} with params {}".format(i, func_vals[-1], model_params[-1]))
            except:
                print("Failed Optimization {}".format(i))

        # set to minimum
        min_index = np.argmin(func_vals)
        self._set_params(model_params[min_index])

    def optimize(self, n_draw=5, max_iters=100, rel_tol=0.5, n_convergence=1000, sampler_burn_in=1000,
                 save_path=None, save_iter=5, **kwargs):
        """
        Optimizes the initiated GP emulator for at most max_iters iterations
        :param n_draw: number of draws in each step
        :param max_iters: maximum number of iterations
        :param rel_tol: relative tolarance of the Bhattacharyya distance, the optimization will stop if the relative
                        change is smaller than rel_tol for 5 consecutive steps
        :param n_convergence: number of sample points for the convergence test
        :param sampler_burn_in: number of burn in steps for the MCMC that draws new samples
        :param save_path: path where to save the intermediate results
        :param save_iter: save itermediate results very so often
        :param kwargs: additional arguments passed to the sample_new routine (e.g. MCMC type etc.)
        """

        if self.objective is None or self.space is None:
            raise ValueError("Iterative optimization is only possible when the GP emulator was initialized with an "
                             "objective function and a design space!")

        # convergence samples
        convergence_samples = self.tfp_prior.sample(n_convergence)

        new_preds = None
        old_coef = None
        changes = []
        for i in range(max_iters):
            old_preds = new_preds

            # get new samples
            t0 = time()
            new_samples = self.sample_new(n_draw, burn_in=sampler_burn_in, **kwargs)
            t1 = time()
            print("Drawn {} new samples in {} sec...".format(n_draw, t1 - t0))

            # eval objective
            t0 = time()
            Y_new, var_new = np.split(self.objective(new_samples), axis=1, indices_or_sections=2)
            t1 = time()
            print("Objective evaluation took {} sec...".format(t1 - t0))

            # normalize
            if self.normalize_Y:
                self.Y_all = self.Y_all * self.Y_std + self.Y_mean
                self.var_estimates *= self.Y_std ** 2
            self.Y_all = np.concatenate([self.Y_all, Y_new], axis=0)
            self.var_estimates = np.concatenate([self.var_estimates, var_new], axis=0)

            if self.normalize_Y:
                self.Y_mean = np.mean(self.Y_all)
                if not self.mean_only:
                    self.Y_std = np.std(self.Y_all)

                self.Y_all = (self.Y_all - self.Y_mean) / self.Y_std
                self.var_estimates /= self.Y_std ** 2

            # stack
            self.params = self.unnormalize_params(self.params, self.rot_mat, self.rot_mean, self.rot_std)
            self.params = np.concatenate([self.params, new_samples], axis=0)
            self.params, self.rot_mat, self.rot_mean, self.rot_std = self.normalize_params(self.params)

            # model
            print("N params: ", len(self.params))
            data = np.concatenate([self.Y_all, np.maximum(self.var_estimates, self.minimum_variance)], axis=1)
            self.model = gpflow.models.VGP((self.params.astype(self.default_np_float),
                                            data.astype(self.default_np_float)),
                                           kernel=self.kern, likelihood=self.likelihood, num_latent_gps=1)

            t0 = time()
            self.optimize_model()
            t1 = time()
            print("GP optimization took {} sec with {} restarts...".format(t1 - t0, self.num_restarts))

            # new acqui
            self.current_transform = lambda x: self.transform_params(x, self.rot_mat, self.rot_mean, self.rot_std)

            # TESTING
            # =======
            # TODO: implement this better
            current_samples = self.transform_params(convergence_samples, self.rot_mat, self.rot_mean, self.rot_std)
            current_samples = current_samples.astype(self.default_np_float)
            # get new preds (unnorm)
            new_preds = np.exp(self.model.predict_f(current_samples)[0].numpy().ravel() * self.Y_std + self.Y_mean)
            if old_preds is not None:
                # Bhattacharyya distance
                new_coef = np.mean(np.sqrt(old_preds * new_preds))
                print(new_coef)
                if old_coef is not None:
                    rel_change = 100 * np.abs(old_coef - new_coef) / new_coef
                    print("Relavitve change: ", rel_change)
                    changes.append(rel_change)
                old_coef = new_coef

                if len(changes) > 5 and not np.any(np.asarray(changes)[-5:] > rel_tol):
                    break

            # Save stuff
            if save_path is not None and i > 0 and i % save_iter == 0:
                current_path = os.path.join(save_path, "iter_%i" % (i))
                if not os.path.exists(current_path):
                    os.mkdir(current_path)
                    self.save_model(current_path)
                    print("Saved intermediate model at <{}>".format(current_path))
                else:
                    print("Save path <{}> exists, "
                          "skipping save model in order to avoid overwriting...".format(current_path))

        print("Done...")

    def sample_new(self, n_draw, burn_in=1000, n_leap=10, step_size=0.05, MCMC_type="Hasting",
                   parallel_iterations=10, start_type="prior", num_results=250, n_chains=None,
                   replace_post=False, min_dist=1e-3, hasting_scale=0.05):
        """
        Draws new samples from the aquisition function
        :param n_draw: number of samples to draw
        :param burn_in: number of burn in steps
        :param n_leap: number of leap frog steps if MCMC_type==HMC
        :param step_size: step size for the HMC algorithm
        :param MCMC_type: type of MCMC (either HMC or Hasting)
        :param parallel_iterations: parallel iterations of the sample chain procedure
        :param start_type: select MCMC initial state by sampling from prior or from the sample of the objective funtion
                           (posterior) weighted by their probability
        :param num_results: number of results to draw (has to be large enough such that accepted > n_draw)
        :param n_chains: number of parallel chains
        :param replace_post: if start_type == posterior, draw with replacement or not
        :param min_dist: minimum distance accepted samples need to be appart to count as "new"
        :param hasting_scale: scale of the Gaussian noise used for the Hastings proposals
        :return: the new samples as numpy array with default float type
        """
        # set number of chains to at least 2*ndim if not defines
        if n_chains is None:
            n_chains = 2 * self.dims

        # starting points
        if start_type == "prior":
            start = self.tfp_prior.sample(n_chains)
        elif start_type == "posterior":
            probs = np.ravel(self.Y_all * self.Y_std + self.Y_mean)
            probs = probs - np.max(probs)
            probs = np.exp(probs)
            param = self.unnormalize_params(self.params, self.rot_mat, self.rot_mean, self.rot_std)
            choices = np.random.choice(a=len(param), size=n_chains, p=probs / np.sum(probs), replace=replace_post)
            start = tf.convert_to_tensor(param[choices], dtype=self.default_tf_float)
        else:
            raise IOError("Unknown start_type: {}".format(start_type))

        # predictor (with alpha value)
        rot_mat = tf.constant(self.rot_mat, dtype=self.default_tf_float)
        rot_mean = tf.constant(self.rot_mean, dtype=self.default_tf_float)
        rot_std = tf.constant(self.rot_std, dtype=self.default_tf_float)
        Y_mean = tf.constant(self.Y_mean, dtype=self.default_tf_float)
        Y_std = tf.constant(self.Y_std, dtype=self.default_tf_float)

        @tf.function
        def log_prob_no_prior(X):
            rot_params = tf.einsum("ij,aj->ai", rot_mat, X)
            X = (rot_params - rot_mean) / rot_std
            mean, var = self.model.predict_f(X)
            if self.normalize_Y:
                mean = mean * Y_std + Y_mean
                var *= Y_std ** 2

            if self.alpha < 1e-8:
                return mean
            else:
                return tf.math.log(tf.exp(mean) + self.alpha * (tf.exp(var) - 1.0) * tf.exp(2 * mean + var))

        # def log prob
        @tf.function
        def log_prob(x):
            # log prob returns -inf if not in prior
            condition = tf.reduce_any(self.tfp_prior.log_prob(x) < -1000, axis=-1)
            if_true = tf.ones_like(condition, dtype=self.default_tf_float) * -np.inf
            if_false = tf.squeeze(log_prob_no_prior(x), axis=-1)
            return tf.where(condition, if_true, if_false)

        if MCMC_type == "HMC":
            kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=log_prob,
                                                    num_leapfrog_steps=n_leap,
                                                    step_size=step_size)
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=kernel, num_adaptation_steps=int(burn_in * 0.8))

            # Run the chain (with burn-in).
            @tf.function
            def run_chain():
                # Run the chain (with burn-in).
                samples, accepted = tfp.mcmc.sample_chain(
                    num_results=num_results,
                    num_burnin_steps=burn_in,
                    current_state=start,
                    kernel=kernel,
                    parallel_iterations=parallel_iterations,
                    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
                return samples, accepted

            samples, accepted = run_chain()


        elif MCMC_type == "Hasting":
            @tf.function
            def run_chain():
                samples, accepted = tfp.mcmc.sample_chain(
                    num_results=num_results,
                    current_state=start,
                    kernel=tfp.mcmc.RandomWalkMetropolis(
                        target_log_prob_fn=log_prob,
                        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=hasting_scale, name=None)),
                    num_burnin_steps=burn_in,
                    # Thinning.
                    num_steps_between_results=1,
                    parallel_iterations=parallel_iterations,
                    trace_fn=lambda _, pkr: pkr.is_accepted)
                return samples, accepted

            samples, accepted = run_chain()

        else:
            raise IOError("Unkown MCMC_type: {}".format(MCMC_type))

        samples = samples.numpy()
        accepted = accepted.numpy()

        n_accepted = np.sum(accepted)
        if self.verbosity > 0:
            print("Accepted {} samples with burn in {} and num results {}".format(n_accepted, burn_in, num_results))
        if n_accepted < n_draw:
            raise ValueError("Not enough samples where accecpted with the given burn in and num result,"
                             "try to increase these values to avoid this error.")

        samples = samples.reshape((-1, self.dims))
        np.random.shuffle(samples)

        # we want n_draw samples that are at least min_dist apart
        new_samples = np.zeros((n_draw, self.dims), dtype=self.default_np_float)
        new_samples[0] = samples[0]
        count = 1
        for samp in samples[1:]:
            if np.all((np.linalg.norm(new_samples - samp, axis=1) > min_dist)[:count]):
                new_samples[count] = samp
                count += 1
            if count == n_draw:
                break

        if count != n_draw:
            raise ValueError("Number of samples whose distance is large than min_dist is less than n_draw. "
                             "Either increase num_results or decrease min_dist!")

        # return n_draws
        return new_samples

    @classmethod
    def normalize_params(self, params):
        """
        normalizes params to unit variance
        """
        # make the params linearly uncorrelated
        cov = np.cov(params, rowvar=False)
        # eigenvals and vecs
        w, v = np.linalg.eig(cov)
        # rot mat is v.T
        rot_mat = v.T
        # dot prod
        rot_params = np.einsum("ij,aj->ai", rot_mat, params)
        # mean
        rot_mean = np.mean(rot_params, axis=0, keepdims=True)
        # std (ddof of np.cov for consistency)
        rot_std = np.std(rot_params, axis=0, keepdims=True, ddof=1)
        # normalize
        new_params = (rot_params - rot_mean) / rot_std

        return new_params, rot_mat, rot_mean, rot_std

    @classmethod
    def transform_params(self, params, rot_mat, rot_mean, rot_std):
        """
        Normalizes params given the rot, shift and scale
        """
        rot_params = np.einsum("ij,aj->ai", rot_mat, params)
        new_params = (rot_params - rot_mean) / rot_std
        return new_params

    @classmethod
    def reverse_transform_params(self, params, rot_mat, rot_mean, rot_std):
        """
        Makes the transformation in reverse
        """
        new_params = params * rot_std + rot_mean
        new_params = np.einsum("ij,aj->ai", rot_mat.T, new_params)

        return new_params

    @classmethod
    def unnormalize_params(self, params, rot_mat, rot_mean, rot_std):
        """
        Removes normalization
        """
        new_params = params * rot_std + rot_mean
        # inverse rotation
        new_params = np.einsum("ij,aj->ai", rot_mat.T, new_params)
        return new_params

    def save_model(self, save_dir):

        # save the kernel
        kern_val = np.array([self.model.kernel.variance.numpy(),
                             self.model.kernel.lengthscales.numpy()])
        q_mu = self.model.q_mu.numpy()
        q_sqrt = self.model.q_sqrt.numpy()
        # save model params
        np.save(os.path.join(save_dir, "kern_" + self.kern_type + ".npy"), kern_val)
        np.save(os.path.join(save_dir, "q_mu.npy"), q_mu)
        np.save(os.path.join(save_dir, "q_sqrt.npy"), q_sqrt)
        np.save(os.path.join(save_dir, "min_var.npy"), np.array([self.minimum_variance], dtype=self.default_np_float))
        # save params
        params = self.unnormalize_params(self.params, self.rot_mat, self.rot_mean, self.rot_std)
        np.save(os.path.join(save_dir, "params.npy"), params)

        # save the evals
        Y_all = self.Y_all * self.Y_std + self.Y_mean
        var_estimates = self.var_estimates * self.Y_std ** 2
        if self.normalize_Y:
            if not self.mean_only:
                path = os.path.join(save_dir, "evals_norm.npy")
            else:
                path = os.path.join(save_dir, "evals_norm_mean_only.npy")
        else:
            path = os.path.join(save_dir, "evals.npy")
        np.save(path, Y_all)
        # we save the var estimates, but they are not needed for the restore
        np.save(os.path.join(save_dir, "var_estimates.npy"), var_estimates)

    def get_noiseless_predictor(self):
        # get the relevant stuff
        Y_std = self.Y_std
        Y_mean = self.Y_mean

        transform = lambda x: self.transform_params(x, self.rot_mat, self.rot_mean, self.rot_std)

        model = self.model

        def noiseless_predictor(X):
            X = transform(X)
            preds = model.predict_f(X)[0].numpy()
            return preds * Y_std + Y_mean

        return noiseless_predictor

    @classmethod
    def restore_noiseless_predictor(self, restore_path, numpy=True):
        # params
        params = np.load(os.path.join(restore_path, "params.npy"))
        params, rot_mat, rot_mean, rot_std = self.normalize_params(params)
        transform = lambda x: self.transform_params(x, rot_mat, rot_mean, rot_std)
        # restore kernel
        if os.path.exists(os.path.join(restore_path, "kern_matern52.npy")):
            kern = np.load(os.path.join(restore_path, "kern_matern52.npy"), allow_pickle=True)
            print("restoring matern52 kernel with variance {} and lengthscale {}...".format(kern[0], kern[1]))
            kernel = gpflow.kernels.Matern52(variance=kern[0], lengthscales=kern[1])
        elif os.path.exists(os.path.join(restore_path, "kern_exponential.npy")):
            kern = np.load(os.path.join(restore_path, "kern_exponential.npy"), allow_pickle=True)
            print("restoring exponential kernel with variance {} and lengthscale {}...".format(kern[0], kern[1]))
            kernel = gpflow.kernels.Exponential(variance=kern[0], lengthscales=kern[1])

        # restore the other model params
        q_mu = np.load(os.path.join(restore_path, "q_mu.npy"))
        q_sqrt = np.load(os.path.join(restore_path, "q_sqrt.npy"))
        var_estimates = np.load(os.path.join(restore_path, "var_estimates.npy"))
        min_variance = np.load(os.path.join(restore_path, "min_var.npy"))[0]

        # restore evals
        if os.path.exists(os.path.join(restore_path, "evals_norm.npy")):
            Y_all = np.load(os.path.join(restore_path, "evals_norm.npy"))
            Y_mean = np.mean(Y_all)
            Y_std = np.std(Y_all)
            Y_all = (Y_all - Y_mean) / Y_std
        elif os.path.exists(os.path.join(restore_path, "evals_norm_mean_only.npy")):
            Y_all = np.load(os.path.join(restore_path, "evals_norm_mean_only.npy"))
            Y_mean = np.mean(Y_all)
            Y_std = 1.0
            Y_all = (Y_all - Y_mean) / Y_std
        else:
            Y_all = np.load(os.path.join(restore_path, "evals.npy"))
            Y_mean = 0.0
            Y_std = 1.0

        # build the model
        likelihood = HeteroskedasticGaussian()
        data = np.concatenate([Y_all, np.maximum(var_estimates, min_variance)], axis=1)
        model = gpflow.models.VGP((params.astype(self.default_np_float),
                                   data.astype(self.default_np_float)),
                                  kernel=kernel, likelihood=likelihood, num_latent_gps=1)

        # assign variables
        model.q_mu.assign(q_mu)
        model.q_sqrt.assign(q_sqrt)

        if numpy:
            def noiseless_predictor(X):
                X = transform(X)
                preds = model.predict_f(X)[0].numpy()
                return preds * Y_std + Y_mean

            return noiseless_predictor

        else:
            if self.default_tf_float is tf.float64:
                print("Warning: tf function should be used with float32")

            rot_mat = tf.constant(rot_mat, dtype=self.default_tf_float)
            rot_mean = tf.constant(rot_mean, dtype=self.default_tf_float)
            rot_std = tf.constant(rot_std, dtype=self.default_tf_float)
            Y_mean = tf.constant(Y_mean, dtype=self.default_tf_float)
            Y_std = tf.constant(Y_std, dtype=self.default_tf_float)

            # tf function
            @tf.function(input_signature=[tf.TensorSpec(shape=(None, params.shape[1]), dtype=self.default_tf_float)])
            def noiseless_predictor(X):
                rot_params = tf.einsum("ij,aj->ai", rot_mat, X)
                X = (rot_params - rot_mean) / rot_std
                preds = model.predict_f(X)[0]
                return preds * Y_std + Y_mean

            return noiseless_predictor
