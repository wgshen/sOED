import numpy as np
from .utils import *
import emcee

class SOED(object):
    """
    A base class for sequential optimal experimental design (sOED). This class
    takes basic inputs (e.g., forward model, reward function, physical state
    transition, and dimensions of the problem, etc.) to construct an sOED 
    framework. However, this class does not include method functions to solve 
    the sOED problem.
    Please refer to https://arxiv.org/abs/1604.08320 (Huan2016) for more details
    of sOED. Note that the results reproduced from this code might be different
    from the results in the paper, because we use different settings in this 
    code (e.g., the exploration policy, the way to estimate the KL divergence, 
    the representation of belief state, the optimization method, the choice
    of features, and the way to estimate the expectation); nevertheless, the 
    performance should be similar.
    This code accommodates continuous unknown parameters with normal or uniform 
    prior, continuous design with upper and lower bounds, and additive Gaussian 
    noise on observations.

    Parameters
    ----------
    model_fun : function
        Forward model function G_k(theta, d_k, x_{k,p}). It will be abbreviated 
        as m_f inside this class.
        The forward model function should take following inputs:
            * theta, numpy.ndarray of size (n_sample or 1, n_param)
                Parameter samples.
            * d, numpy.ndarray of size (n_sample or 1, n_design)
                Designs.
            * xp, numpy.ndarray of size (n_sample or 1, n_phys_state)
                Physical states.
        and the output is 
            * numpy.ndarray of size (n_sample, n_obs).
        When the first dimension of theta, d or xp is 1, it should be augmented
        to align with the first dimension of other inputs (i.e., we reuse it for
        all samples).
    n_stage : int
        Number of experiments to be designed and conducted.
    n_param : int
        Dimension of parameter space, should not be greater than 3 in this
        version because we are using grid discritization on parameter sapce.
    n_design : int
        Dimension of design space.
    n_obs : int
        Dimension of observation space.
    prior_info : list, tuple or numpy.ndarray of size (n_param, 3)
        It includes the information of the prior. In this version, we only 
        allow to use independent normal or uniform distributions on each 
        dimension of parameter space. In the future version, we will let users 
        provide their owe functions to generate samples from the prior and 
        evaluate prior PDFs. 
        The length of prior_info should be n_param. k-th entry of prior_info 
        includes the following three components for k-th dimension of the 
        paramemter (could be list, tuple or numpy.ndarray in the following 
        ordering):
            * prior_type : str
                Type of prior of k-th parameter.
                "uniform" indicates uniform distribution.
                "normal" or "gaussian" indicates normal distribution.
            * prior_loc : float or int
                Mean for normal, or left bound for uniform.
            * prior_scale : float or int
                Std for normal, or range for uniform.
    design_bounds : list, tuple or numpy.ndarray of size (n_design, 2)
        It includes the constraints of the design variable. In this version, we
        only allow to set hard limit constraints. In the future version, we may
        allow users to provide their own constraint function.
        The length of design_bounds should be n_design.
        k-th entry of design_bounds is a list, tuple or numpy.ndarray like 
        (lower_bound, upper_bound) for the limits of k-th design variable.
    noise_info : list, tuple or numpy.ndarray of size (n_obs, 3)
        It includes the statistics of additive Gaussian noise.
        The length of noise_info should be n_obs. k-th entry of noise_info is a 
        list, tuple or numpy.ndarray including
            * noise_loc : float or int
            * noise_base_scale : float or int
                It will be abbreviated as noise_b_s in this class.
            * noise_ratio_scale : float or int
                It will be abbreviated as noise_r_s in this class.
        The corresponding noise will follow a gaussian distribution with mean
        noise_loc, std (noise_base_scale + noise_ratio_scale * abs(G)).
    reward_fun : function, optional(default=None)
        User-provided non-KL-divergence based reward function 
        g_k(x_k, d_k, y_k). It will be abbreviated as nlkd_rw_f inside this 
        class.
        The reward function should take following inputs:
            * stage : int
                The stage index of the experiment.
            * xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1)
                Grid discritization of the belief state.
            * xp : np.ndarray of size (n_phys_state)
                The physical state.
            * d : np.ndarray of size (n_design)
                The design variable.
            * y : np.ndarray of size (n_obs)
                The observation.
        and the output is 
            * A float which is the reward.
        Note that the information gain is computed within this class, and does
        not needed to be included in reward_fun.
        When reward_fun is None, the stage reward would be 0, only KL divergence
        from the prior to the posterior will be considered.
    phys_state_info : list, tuple or numpy.ndarray of size (3), 
                      optional(default=None)
        When phy_state_info is None, then there is no physical states in this 
        sOED problem, otherwise it includes the following information of 
        physical states:
            * n_phys_state : int
                Dimension of physical state.
                It will be abbreviated as n_xp inside this class.
            * init_phys_state : list, or tuple
                Initial physical states.
                It will be abbreviated as init_xp inside this class.
                The length of init_phys_state should n_phys_state.
                In the future, we will let phys_state_fun provide the initial 
                physical state, such that it could be stochastic.
            * phys_state_fun : function
                Function to update physical state.
                x_{k+1,p} = phys_state_fun(x_{k,p}, d_k, y_k).
                It will be abbreviated as xp_f inside this class.
                The physical state transition function should take following 
                inputs:
                    * xp : np.ndarray of size (n_sample or 1, n_phys_state)
                        The old physical state before conducting stage-th 
                        experiement.
                    * stage : int
                        The stage index of the experiment.
                    * d : np.ndarray of size (n_sample or 1, n_design)
                        The design variables at stage-th experiment.
                    * y : np.ndarray of size (n_sample or 1, n_obs)
                        The observations at stage-th expriments.
                and the output is 
                    * numpy.ndarray of size (n_sample, n_xp)
                Note that the update of belief state is realized in this class, 
                and does not need to be provided by users.
    n_grid : int, optional(default=50)
        Number of grid points to discretize each dimension of parameter space
        to store the belief state. Using grid discretization is only practical 
        when the dimension is not bigger than 3. 
        In the future version, we plan to use other techniques (MCMC, trasport
        map) to represent the posterior distribution, and n_grid will pribably
        be discarded.
    post_rvs_method : str, optional(default="MCMC")
        Method to sample from the posterior, including:
            * "MCMC", Markov chain Monte Carlo via emcee.
            * "Rejection", rejection sampling, only allowed for 1D parameter.
    random_state : int, optional(default=None)
        It is used as the random seed.        

    Methods
    -------
    prior_logpdf(), prior_pdf()
        Evaluate the prior logpdf (pdf) of parameter samples.
    prior_rvs()
        Generate samples from the prior.
    post_logpdf(), post_pdf()
        Evaluate the posterior logpdf (pdf) of parameter samples.
    post_rvs()
        Generate samples from the posterior.
    xb_f()
        Update the belief state with a single observation and design.
    get_xb()
        Update the belief state with a sequence of observations and designs.
    get_xp()
        Update the physical state with a sequence of observations and designs.
    get_reward()
        Get the reward at a given stage with given state, design and 
        observation.
    get_total_reward()
        Get the total reward give a sequence of designs and observations.

    Future work
    -----------
    Use MCMC or transport map to replace the grid discretization.
    Let users provide their own prior sample generator and prior PDF evaluator.
    Let users provide more complex constraints on design variables.
    Let users provide their own measurement noise function.
    Consider random initial physical state.
    Use underscore to make variables not directly accessible by users, like
    "self._n_stage", and use @property to make it indirectly accessible.
    """
    def __init__(self, model_fun, 
                 n_stage, n_param, n_design, n_obs, 
                 prior_info, design_bounds, noise_info, 
                 reward_fun=None, phys_state_info=None,
                 n_grid=50, post_rvs_method="MCMC", random_state=None):
        NoneType = type(None)
        assert isinstance(random_state, (int, NoneType)), (
               "random_state should be an integer or None.")
        np.random.seed(random_state)
        
        assert callable(model_fun), (
               "model_fun should be a function.")
        self.m_f = model_fun
        self.model_fun = self.m_f

        assert isinstance(n_stage, int) and n_stage > 0, (
               "n_stage should be an integer greater than 0.")
        self.n_stage = n_stage
        assert isinstance(n_param, int) and n_param > 0 and n_param < 4, (
               "n_param should be an integer greater than 0, smaller than 4.")
        self.n_param = n_param
        assert isinstance(n_design, int) and n_design > 0, (
               "n_design should be an integer greater than 0.")
        self.n_design = n_design
        assert isinstance(n_obs, int) and n_obs > 0, (
               "n_obs should be an integer greater than 0.")
        self.n_obs = n_obs
        
        assert isinstance(prior_info, (list, tuple, np.ndarray)), (
               "prior_info should be a list, tuple or numpy.ndarray of size "
               "(n_param, 3).")
        assert len(prior_info) == n_param, (
               "Length of prior_info should equal n_param.")
        uniform_idxs, uniform_lows, uniform_highs = [], [], []
        normal_idxs, normal_locs, normal_scales = [], [], []
        for i in range(n_param):
            assert len(prior_info[i]) == 3, (
                   "Each entry of prior_info is of size 3, including "
                   "prior_type, prior_loc and prior_scale.")
            prior_type, prior_loc, prior_scale = prior_info[i]
            assert prior_type in ("normal", "gaussian", "uniform"), (
                   "{}-th prior_type should be normal(gaussian) or uniform.".
                   format(i))
            assert isinstance(prior_loc, (int, float)), (
                   "{}-th prior_loc should be a number.".format(i))
            assert isinstance(prior_scale, (int, float)), (
                   "{}-th prior_scale should be a number.".format(i))
            if prior_type == "uniform":
                uniform_idxs.append(i)
                uniform_lows.append(prior_loc)
                uniform_highs.append(prior_loc + prior_scale)
            else:
                normal_idxs.append(i)
                normal_locs.append(prior_loc)
                normal_scales.append(prior_scale)
        uniform_lows = np.array(uniform_lows)
        uniform_highs = np.array(uniform_highs)
        normal_locs = np.array(normal_locs)
        normal_scales = np.array(normal_scales)
        # In the future version, prior sample generator and prior PDF evaluator
        # will be provided by users.
        # Prior PDF evaluator.
        # prior_samples is of size (n_sample, n_para).
        # Output is of size (n_sample).
        def prior_logpdf(prior_samples, 
                         uniform_idxs, uniform_lows, uniform_highs,
                         normal_idxs, normal_locs, normal_scales):
            uniform_logpdfs = uniform_logpdf(prior_samples[:, uniform_idxs],
                                             uniform_lows, uniform_highs)
            normal_logpdfs = norm_logpdf(prior_samples[:, normal_idxs],
                                         normal_locs, normal_scales)
            return uniform_logpdfs + normal_logpdfs
        self.prior_logpdf = lambda x: prior_logpdf(x, 
                                                   uniform_idxs, uniform_lows, 
                                                   uniform_highs, normal_idxs,
                                                   normal_locs, normal_scales)
        self.prior_pdf = lambda x: np.exp(self.prior_logpdf(x))
        # Prior sample generator.
        # Output is of size (n_sample, n_param).
        def prior_rvs(n_sample, n_param,
                      uniform_idxs, uniform_lows, uniform_highs,
                      normal_idxs, normal_locs, normal_scales):
            prior_samples = np.zeros((n_sample, n_param))
            uniform_samples = np.random.uniform(uniform_lows,
                                                uniform_highs,
                                                (n_sample, len(uniform_idxs)))
            normal_samples = np.random.normal(normal_locs,
                                              normal_scales,
                                              (n_sample, len(normal_idxs)))
            prior_samples[:, uniform_idxs] = uniform_samples
            prior_samples[:, normal_idxs] = normal_samples
            return prior_samples
        self.prior_rvs = lambda x: prior_rvs(x, n_param,
                                             uniform_idxs, uniform_lows, 
                                             uniform_highs, normal_idxs,
                                             normal_locs, normal_scales)

        assert isinstance(design_bounds, (list, tuple, np.ndarray)), (
               "design_bounds should be a list, tuple or numpy.ndarray of " 
               "size (n_design, 2).")
        assert len(design_bounds) == n_design, (
               "Length of design_bounds should equal n_design.")
        for i in range(n_design):
            assert len(design_bounds[i]) == 2, (
                   "Each entry of prior_info is of size 2, including "
                   "lower bound and upper bound.")
            l_b, u_b = design_bounds[i]
            assert isinstance(l_b, (int, float)), (
                   "{}-th lower bound should be a number.".format(i))
            assert isinstance(u_b, (int, float)), (
                   "{}-th upper_bound should be a number.".format(i))
        # size (n_design, 2)
        self.design_bounds = np.array(design_bounds)

        assert isinstance(noise_info, (list, tuple, np.ndarray)), (
               "noise_info should be a list, tuple or numpy.ndarray of " 
               "size (n_obs, 3).")
        assert len(noise_info) == n_obs, (
               "Length of noise_info should equal n_obs.")
        for i in range(n_obs):
            assert len(noise_info[i]) == 3, (
                   "Each entry of noise_info is of size 3, including "
                   "noise_loc, noise_base_scale and noise_ratio_scale.")
            noise_loc, noise_b_s, noise_r_s = noise_info[i]
            assert isinstance(noise_loc, (int, float)), (
                   "{}-th noise_loc should be a number.".format(i))
            assert isinstance(noise_b_s, (int, float)), (
                   "{}-th noise_base_scale should be a number.".format(i))
            assert isinstance(noise_r_s, (int, float)), (
                   "{}-th noise_ratio_scale should be a number.".format(i))
            assert noise_b_s ** 2 + noise_r_s ** 2 > 0, (
                   "Either {}-th noise_base_scale or noise_ratio_scale "
                   "should be greater than 0.".format(i))
        # size (n_obs, 3)
        self.noise_info = np.array(noise_info)
        self.noise_loc = self.noise_info[:, 0]
        self.noise_b_s = self.noise_info[:, 1]
        self.noise_r_s = self.noise_info[:, 2]

        # Non-KL-divergence based reward function
        if reward_fun is None:
            self.nkld_rw_f = lambda *args, **kws: 0
        else:
            assert callable(reward_fun), (
                   "reward_fun should be a function.")
            self.nkld_rw_f = reward_fun

        if phys_state_info is None:
            self.n_xp = 0
            self.init_xp = ()
            self.xp_f = lambda *args, **kws: np.array([])
        else:
            assert (isinstance(phys_state_info, (list, tuple, np.ndarray))
                    and len(phys_state_info) == 3), (
               "phys_state_info should be a list, tuple or numpy.ndarray of "
               "length 3, including n_phys_state, init_phys_state and "
               "phys_state_fun.")
            n_xp, init_xp, xp_f = phys_state_info
            assert isinstance(n_xp, int) and n_xp >= 0, (
                   "n_phys_state should be a non-negative interger.")
            self.n_xp = n_xp
            assert (isinstance(init_xp, (list, tuple, np.ndarray))
                    and len(init_xp) == n_xp), (
                   "init_phys_state should be a list, tuple or numpy.ndarray "
                   "of size n_phys_state.")
            self.init_xp = tuple(init_xp)
            assert callable(xp_f), (
                   "phys_state_fun should be a function.")
            self.xp_f = xp_f
        self.n_phys_state = self.n_xp
        self.init_phys_state = self.init_xp
        self.phys_state_fun = self.xp_f

        assert (isinstance(post_rvs_method, str) 
                and post_rvs_method in ("MCMC", "Rejection")), (
                "post_rvs_method should be either 'MCMC' or 'Rejection'."
                )
        self.post_rvs_method = post_rvs_method

        assert isinstance(n_grid, int) and n_grid >= 5, (
               "n_grid should be an integer greater than 4.")
        self.n_grid = n_grid
        # Set the limit of grid discritization, will be discarded in the future.
        self.param_lim = np.zeros((n_param, 2))
        self.param_lim[uniform_idxs] = np.c_[uniform_lows, uniform_highs]
        self.param_lim[normal_idxs] = np.c_[normal_locs - 5 * normal_scales,
                                            normal_locs + 5 * normal_scales]
        # Initial belief state.
        self.init_xb = self.get_xb(None)
    
    def post_logpdf(self, thetas, 
                    stage=0, d_hist=None, y_hist=None, 
                    xp=None, xp_hist=None, 
                    include_prior=True):
        """
        A function to compute the log-probability of unnormalized posterior  
        after observing a sequence of observations "y_hist" by conducting 
        experiments under designs "d_hist" starting from stage "stage" with 
        intial physical state "xp".
        If "xp_hist" is provided, then we don't need to run the physical state 
        function in this function.

        Parameters
        ----------
        thetas : numpy.ndarray of size (n_sample, n_param)
            The parameter samples whose log-posterior are required.
        stage : int, optional(default=0)
            The starting stage. 0 <= stage <= n_stage.
        d_hist : numpy.ndarray of size (n, n_design), optional(default=None)
            The sequence of designs from stage "stage". n + stage <= n_stage.
        y_hist : numpy.ndarray of size (n, n_obs), optional(default=None)
            The sequence of observations from stage "stage". 
        xp : numpy.ndarray of size (n_phys_state), optional(default=None)
            The physical state at stage "stage".
        xp_hist : numpy.ndarray of size (n_stage + 1, n_phys_state),
                  optional(default=None)
            The historical physical states. This will be helpful if evaluating
            physical state function is expensive.
        include_prior : bool, optional(default=True)
            Include the prior in the posterior or not. It not included, the
            posterior is just a multiplication of likelihoods.

        Returns
        -------
        A numpy.ndarray of size (n_sample) which are log-posteriors.
        """
        if d_hist is None: d_hist = [] 
        if y_hist is None: y_hist = [] 
        if xp is None: xp = np.array(self.init_xp)
        assert (len(d_hist) == len(y_hist) 
                and len(d_hist) + stage <= self.n_stage)
        if len(d_hist) == 0:
            if include_prior:
                return self.prior_logpdf(thetas)
            else:
                return np.zeros(len(thetas))
        else:
            G = self.m_f(stage, thetas, d_hist[0:1], xp.reshape(1, self.n_xp))
            loglikeli = norm_logpdf(y_hist[0:1], 
                                    G + self.noise_loc,
                                    self.noise_b_s + self.noise_r_s * np.abs(G))
            if xp_hist is not None:
                next_xp = xp_hist[stage + 1]
            else:
                next_xp = self.xp_f(xp.reshape(1, self.n_xp), 
                                    stage, 
                                    d_hist[0].reshape(1, -1), 
                                    y_hist[0].reshape(1, -1)).reshape(-1)
            return loglikeli + self.post_logpdf(thetas, 
                                                stage + 1,
                                                d_hist[1:],
                                                y_hist[1:],
                                                next_xp,
                                                xp_hist,
                                                include_prior)

    def post_pdf(self, *args, **kws):
        return np.exp(self.post_logpdf(*args, **kws))

    def post_rvs(self, n_sample, 
                 d_hist=None, y_hist=None, xp_hist=None,
                 use_MCMC=True):
        """
        A function to generate samples from the posterior distribution,
        after observing a sequence of observations 'y_hist' by conducting 
        experiments under designs 'd_hist' from stage 0.
        If "xp_hist" is provided, then we don't need to run the physical state 
        function in this function.

        Parameters
        ----------
        n_sample : int
            Number of posterior samples we want to generate.
        d_hist : numpy.ndarray of size (n, n_design), optional(default=None)
            The sequence of designs from stage 0. n <= n_stage.
        y_hist : numpy.ndarray of size (n, n_obs), optional(default=None)
            The sequence of observations from stage 0. 
        xp_hist : numpy.ndarray of size (n_stage + 1, n_phys_state),
                  optional(default=None)
            The historical physical states. This will be helpful if evaluating
            physical state function is expensive.
        use_MCMC : bool, optional(default=True)
            Whether use MCMC or rejection sampling. use_MCMC=False is only 
            allowed for 1D parameter space. In experience, rejection sampling
            for a not very sharp posterior could be several times faster than
            MCMC in 1D, because MCMC needs to work sequentially, while rejection
            sampling can take advantage of vectorization or parallelization.

        Returns
        -------
        A numpy.ndarray of size (n_sample, n_param).
        """
        if self.post_rvs_method == "MCMC":
            log_prob = lambda x : self.post_logpdf(x.reshape(-1, self.n_param), 
                                                   stage=0,
                                                   d_hist=d_hist,
                                                   y_hist=y_hist,
                                                   xp_hist=xp_hist)
            n_dim, n_walkers = self.n_param, 2 * self.n_param
            theta0 = self.prior_rvs(n_walkers)
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
            sampler.run_mcmc(theta0, 
                             int(n_sample // n_walkers * 1.2), 
                             progress=False)
            return sampler.get_chain().reshape(-1, self.n_param)[-n_sample:]
        elif self.post_rvs_method == "Rejection":
            assert self.n_param == 1, "use_MCMC=False only allowed for 1D case."
            thetas_post = np.empty((0, self.n_param))
            xb = self.get_xb(d_hist=d_hist, y_hist=y_hist, 
                             xp_hist=xp_hist, normalize=False)
            max_pdf = (xb[:, -1] / self.prior_pdf(xb[:, :-1])).max() * 1.2
            while (len(thetas_post) < n_sample):
                thetas_prior = self.prior_rvs(n_sample)
                prior_pdfs = self.prior_pdf(thetas_prior)
                post_pdfs = self.post_pdf(thetas_prior, d_hist=d_hist, 
                                          y_hist=y_hist, xp_hist=xp_hist)
                us = np.random.rand(n_sample)
                accept_idxs = us < (post_pdfs / max_pdf / prior_pdfs)
                thetas_post = np.r_[thetas_post, thetas_prior[accept_idxs]]
            return thetas_post[:n_sample]

    def get_xb(self, xb=None, 
               stage=0, d_hist=None, y_hist=None, 
               xp=None, xp_hist=None,
               normalize=True):
        """
        A function to update the belief state (PDFs on grid discritization) 
        after observing a sequence of observations "y_hist" by conducting 
        experiments under designs "d_hist" starting from stage "stage" with 
        intial physical state "xp".
        If "xp_hist" is provided, then we don't need to run the physical state 
        function in this function.

        Parameters
        ----------
        xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1),
             optional(default=None)
            Grid discritization of old belief state.
            In the second dimension, the first n_param components are locations
            of the grid, and the last component is the corresponding PDF value.
        stage : int, optional(default=0)
            The starting stage. 0 <= stage <= n_stage.
        d_hist : numpy.ndarray of size (n, n_design), optional(default=None)
            The sequence of designs from stage "stage". n + stage <= n_stage.
        y_hist : numpy.ndarray of size (n, n_obs), optional(default=None)
            The sequence of observations from stage "stage". 
        xp : numpy.ndarray of size (n_phys_state), optional(default=None)
            The physical state at stage "stage".
        xp_hist : numpy.ndarray of size (n_stage + 1, n_phys_state),
                  optional(default=None)
            The historical physical states. This will be helpful if evaluating
            physical state function is expensive.
        normalize : bool, optional(default=True)
            Whether normalize the PDF values.

        Returns
        -------
        A numpy.ndarray of size (n_grid ** n_param, n_param + 1) which is the
        new belief state.
        """
        if xb is None:
            grids = np.zeros((self.n_param, self.n_grid))
            for i in range(self.n_param):
                grids[i] = np.linspace(self.param_lim[i, 0], 
                                       self.param_lim[i, 1], 
                                       self.n_grid)
            dgrid = np.prod(grids[:, 1] - grids[:, 0])
            self.dgrid = dgrid
            # xb = np.array(np.meshgrid(*grids, indexing='ij')).T
            xb = np.stack((np.meshgrid(*grids, indexing='ij')), 
                axis=self.n_param)
            xb = xb.reshape((self.n_grid ** self.n_param, self.n_param))
            xb = np.c_[xb, self.prior_pdf(xb)]
        new_xb = np.copy(xb)
        new_xb[:, -1] *= self.post_pdf(new_xb[:, 0:-1],
                                       stage=stage,
                                       d_hist=d_hist,
                                       y_hist=y_hist,
                                       xp=xp,
                                       xp_hist=xp_hist,
                                       include_prior=False)
        if normalize:
            new_xb[:, -1] /= new_xb[:, -1].sum() * self.dgrid
        return new_xb

    def xb_f(self, xb=None, 
             stage=0, d=None, y=None, 
             xp=None, xp_hist=None):
        """
        A function to update the belief state (PDFs on grid discritization) 
        for one step after observing an observation 'y' by conducting the
        "stage"-th experiment under design 'd' with physical state "xp".
        If "xp_hist" is provided, then we don't need to run the physical state 
        function in this function.

        Parameters
        ----------
        xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1),
             optional(default=None)
            Grid discritization of old belief state.
            In the second dimension, the first n_param components are locations
            of grid, and the last component is the PDF value.
        stage : int, optional(default=0)
            The starting stage. 0 <= stage <= n_stage.
        d : numpy.ndarray of size (n_design), optional(default=None)
            The sequence of designs from stage "stage". n + stage <= n_stage.
        y : numpy.ndarray of size (n, n_obs), optional(default=None)
            The sequence of observations from stage "stage". 
        xp : numpy.ndarray of size (n_phys_state), optional(default=None)
            The physical state at stage "stage".
        xp_hist : numpy.ndarray of size (n_stage + 1, n_phys_state)
                  , optional(default=None)
            The historical physical states. This will be helpful if evaluating
            physical state function is expensive.

        Returns
        -------
        A numpy.ndarray of size (n_grid ** n_param, n_param + 1) which is the
        new belief state.
        """
        if d is None: d = np.array([])
        if y is None: y = np.array([])
        return self.get_xb(xb, stage, 
                           d.reshape(1, -1), y.reshape(1, -1), 
                           xp, xp_hist)

    def get_xp(self, xp=None,
               stage=0, d_hist=None, y_hist=None):
        """
        A function to update the physical state after observing a sequence of 
        observations "y_hist" by conducting experiements under designs "d_hist"
        starting from stage "stage" with initial physical state "xp".

        Parameters
        ----------
        xp : numpy.ndarray of size (n_phys_state), optional(default=None)
            The old physical state at stage "stage".
        stage : int, optional(default=0)
            The starting stage. 0 <= stage <= n_stage.
        d_hist : numpy.ndarray of size (n, n_design), optional(default=None)
            The sequence of designs from stage "stage". n + stage <= n_stage.
        y_hist : numpy.ndarray of size (n, n_obs), optional(default=None)
            The sequence of observations from stage "stage". 

        Returns
        -------
        A numpy.ndarray of size (n_phys_state) which is the new physical state.
        """
        if xp is None:
            new_xp = np.array(self.init_xp)
        else:
            new_xp = np.copy(xp)
        if d_hist is None: d_hist = np.array([])
        if y_hist is None: y_hist = np.array([])
        for k in range(len(d_hist)):
            new_xp = self.xp_f(new_xp.reshape(1, -1), 
                               stage + k,
                               d_hist[k].reshape(1, -1), 
                               y_hist[k].reshape(1, -1)).reshape(-1)
        return new_xp

    def get_reward(self, stage=0, xb=None, xp=None, d=None, y=None):
        """
        A function to compute the reward at given "stage", with belief state 
        "xb", physical state "xp", design 'd' and observation 'y'.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage.
        xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1), 
             optional(default=None)
            The belief state at stage "stage". 
        xp : numpy.ndarray of size (n_phys_state), optional(default=None)
            The physical state at stage "stage".
        d : numpy.ndarray of size (n_design), optional(default=None)
            The design variable at stage "stage".
        y : numpy.ndarray of size (n_obs), optional(default=None)
            The observation at stage "stage". 

        Returns
        -------
        A float value which is the reward.
        """
        assert stage >= 0 and stage <= self.n_stage
        nkld_reward = self.nkld_rw_f(stage, xb, xp, d, y)
        kld = 0
        if stage == self.n_stage and xb is not None:
            kld = np.sum(xb[:, -1] * 
                         (np.log(xb[:, -1] + 1e-32) - 
                          np.log(self.init_xb[:, -1] + 1e-32))) * self.dgrid
        return nkld_reward + kld

    def get_total_reward(self, d_hist, y_hist, return_reward_hist=False):
        """
        A function to compute the total reward of a given sequence of designs 
        "d_hist" and observations "y_hist".

        Parameters
        ----------
        d_hist : numpy.ndarray of size (n_stage, n_design), 
                 optional(default=None)
            The sequence of designs from stage 0.
        y_hist : numpy.ndarray of size (n_stage, n_obs), 
                 optional(default=None)
            The sequence of observations from stage 0. 

        Returns
        -------
        A float value which is the total reward.
        (optional) A numpy.ndarray of size (n_stage + 1) which are rewards in 
        each stage.
        """
        assert len(d_hist) == len(y_hist) and len(d_hist) == self.n_stage
        xb = np.copy(self.init_xb)
        xp = np.array(self.init_xp)
        reward_hist = np.zeros(self.n_stage + 1)
        for k in range(self.n_stage + 1):
            if k < self.n_stage:
                reward_hist[k] = self.get_reward(k, xb, xp, 
                                                 d_hist[k], y_hist[k])
                xb = self.xb_f(xb, k, d_hist[k:k + 1], y_hist[k:k + 1], xp)
                xp = self.xp_f(xp.reshape(1, -1), 
                               k, 
                               d_hist[k].reshape(1, -1), 
                               y_hist[k].reshape(1, -1)).reshape(-1)
            else:
                reward_hist[k] = self.get_reward(k, xb, xp, None, None)
        if return_reward_hist:
            return reward_hist.sum(), reward_hist
        else:
            return reward_hist.sum()              