import numpy as np
from .soed import SOED
from .utils import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class ADPsOED(SOED):
    """
    A class for solving sequential optimal experimental design (sOED) problems 
    using approximate dynamic programming (ADP) method. This class is based on
    the SOED class, and realize the method functions of ADP upon it.
    Please refer to https://arxiv.org/abs/1604.08320 (Huan2016) for more details
    of sOED and ADP.
    Due to the use of parametric feature-based linear fit of value function,
    this code only works for 1D parameter space (i.e., n_param=1).
    We also restrict the design space to be 1D in this code (i.e., n_design=1).
    For problems with higher design and parameter spaces, please use policy 
    gradient based method (PGsOED).

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
    n_basic_feature : int, optional(default=2)
        The number of basic features we use to represent the belief state. Note 
        that the physical state will be appended to the features of belief
        state, so the total number of basic features is (n_basic_feature + 
        n_phys_state). Also, 1 <= n_basic_feature <= 4. Basic features of belief 
        state includes
            * mean
            * log(var)
            * var
            * log(4-th moment)
    n_degree : int, optional(default=2)
        The degree of polynomial combinations of basic features we will use in 
        the linear fit of value function.
        For example, if n_degree=2, the total number of features in the linear 
        fit would be ((1 + n_basic_feature + n_phys_state) ** 2 + 
                      (1 + n_basic_feature + n_phys_state)) // 2. 
        Add 1 is because we include the bias (intercept) term.

    Methods
    -------
    load_feature_weights()
        Load the weights of features for the linear fit of value function.
    get_feature_weights()
        Return the weights of features.
    get_feature()
        Return the features for the linear fit of value function.
    one_step_aheah_reward()
        Compute the one step lookahead reward.
    expect_on_obs()
        Estimate the expectation of one step lookahead reward on observations
        given state and design.
    optimize()
        Find the optimum that maximizes the objective function.
    get_design()
        Return the current optimal design given historical designs and 
        observations.
    simulate_trajs()
        Simulating trajectories by following some certain rules, could be either
        exploration or exploitation.
    approx_value_iter()
        Approximte value iteration to update the feature weights.
    soed()
        Run exploration, exploitation and approximate value iteration for give
        udpates.
    asses()
        Asses the performance of current policy.

    Future work
    -----------
    """
    def __init__(self, model_fun, 
                 n_stage, n_param, n_design, n_obs, 
                 prior_info, design_bounds, noise_info, 
                 reward_fun=None, phys_state_info=None,
                 n_grid=50, post_rvs_method="MCMC", random_state=None,
                 n_basic_feature=2, n_degree=2):
        super().__init__(model_fun, n_stage, n_param, n_design, n_obs, 
                         prior_info, design_bounds, noise_info, 
                         reward_fun, phys_state_info,
                         n_grid, post_rvs_method, random_state)
        assert n_param == 1, "This code only works for 1D parameter space."
        assert n_design == 1, "This code only works for 1D design space."
        assert (isinstance(n_basic_feature, int) 
                and n_basic_feature >= 1 
                and n_basic_feature <= 4), (
               "n_feature should be an integer between 1 and 4, inclusively.")
        self.n_basic_feature = n_basic_feature
        assert (isinstance(n_degree, int) 
                and n_degree >= 1 
                and n_degree <= 3), (
               "n_degree should be an integer between 1 and 3, inclusively.")
        self.n_degree = n_degree
        # Get the number of features in the linear fit of value function.
        poly = PolynomialFeatures(degree=self.n_degree)
        basic_features = np.zeros((1, n_basic_feature + self.n_xp))
        features = poly.fit_transform(basic_features)[0]
        self.n_feature = len(features)
        # Set the update level to 0.
        self.update = 0
        # Initialize the feature weights to be 0.
        # Note that we only need the feature weights from stage 1 to 
        # n_stage - 1.
        self.w_feature = np.zeros((self.n_stage - 1, self.n_feature))

    def load_feature_weights(self, w_feature):
        """
        Load the weights of features.
        """
        assert w_feature.shape == (self.n_stage - 1, self.n_feature)
        self.w_feature = np.copy(w_feature)
        self.update = 1

    def get_feature_weights(self):
        """
        Get the weights of features.
        """
        print("Return the feature weights from stage 1 to stage", 
              self.n_stage - 1)
        return self.w_feature

    def get_features(self, xb, xp):
        """
        Get the features of linear fit of value function.

        Parameters
        ----------
        xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1)
            Grid discritization of the belief state.
        xp : numpy.ndarray of size (n_phys_state)
            The physical state.

        Returns
        -------
        A numpy.ndarray of size (n_feature) represents the features.
        """
        basic_features = np.zeros(self.n_basic_feature + self.n_xp)
        basic_features[self.n_basic_feature:] = xp
        mean = np.sum(xb[:, 0] * xb[:, 1] * self.dgrid)
        logvar = np.log(np.sum((xb[:, 0] - mean) ** 2 * 
                               xb[:, 1] * 
                               self.dgrid))
        var = np.exp(logvar)
        logm4 = np.log(np.sum((xb[:, 0] - mean) ** 4 * 
                              xb[:, 1] * 
                              self.dgrid))
        basic_features[0:2] = np.array([mean, logvar])
        if self.n_basic_feature>2: basic_features[2] = var
        if self.n_basic_feature>3: basic_features[3] = logm4
        basic_features = basic_features.reshape(1, -1)
        poly = PolynomialFeatures(degree=self.n_degree)
        return poly.fit_transform(basic_features)[0]

    def one_step_aheah_reward(self, stage, xb, xp, d, y):
        """
        A function to compute the one step lookahead reward, which is 
        g_k(x_k, d_k, y_k) + tilde(J)_{k+1}(F_k(x_k, d_k, y_k)) in Equation 12
        of Huan2016.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage.
        xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1)
            The belief state at stage "stage".
        xp : numpy.ndarray of size (n_phys_state)
            The physical state at stage "stage".
        d : numpy.ndarray of size (n_design)
            The design variable at stage "stage".
        y : numpy.ndarray of size (n_obs)
            The observation at stage "stage". 

        Returns
        -------
        A flaot which is the one step lookahead reward (g + J).
        """
        assert stage >= 0 and stage < self.n_stage
        # Get g_k(x_k, d_k, y_k).
        g = self.get_reward(stage, xb, xp, d, y)
        # Get new belief state and physical state.
        new_xb = self.xb_f(xb, stage, d, y, xp)
        new_xp = self.xp_f(xp.reshape(1, -1), 
                           stage, 
                           d.reshape(1, -1), 
                           y.reshape(1, -1)).reshape(-1)
        # If next stage is terminal stage, then we can get the one step 
        # lookahead reward just by computing the terminal reward.
        if stage == self.n_stage - 1:
            J = self.get_reward(stage + 1, new_xb, new_xp, None, None)
        # Otherwise, we need to approximate the one step lookahead reward
        # using the linear dot between features and feature weights.
        else:
            # Get features corresponding to the new state.
            features = self.get_features(new_xb, new_xp)
            # Estimate tilde(J)_{k+1}(F_k(x_k, d_k, y_k)).
            J = np.dot(features, self.w_feature[stage])
        return g + J
        
    def expect_on_obs(self, stage, xb, xp, d, d_hist, y_hist, xp_hist=None):
        """
        A function to estimate
        E_{y_k | x_k, d_k}[g_k(x_k, d_k, y_k) 
                           + tilde(J)_{k + 1}(F_k(x_k, d_k, y_k))]
        in Equation 12 of Huan2016. 
        The epectation is estimated via MCMC samples.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1)
            The belief state at stage "stage".
        xp : numpy.ndarray of size (n_phys_state)
            The physical state at stage "stage".
        d : numpy.ndarray of size (n_design)
            The design variable at stage "stage".
        d_hist : numpy.ndarray of size (stage, n_design)
            The sequence of designs before stage "stage".
        y_hist : numpy.ndarray of size (stage, n_obs)
            The sequence of observations before stage "stage". d_hist and y_hist
            are used to sample parameters from the belief state.
        xp_hist : numpy.ndarray of size (n_stage + 1, n_phys_state), 
                  optional(default=None)
            The historical physical states. This will be helpful if evaluating
            physical state function is expensive.

        Returns
        -------
        A float value which is the estimated expectation.
        """
        # Number of observation samples we want to generate to estimate the 
        # expectation.
        n_sample = self.n_param * 100
        # Sample parameter samples from the belief state.
        thetas = self.post_rvs(n_sample, d_hist, y_hist, xp_hist)
        # Run the forward model to get corresponding observation samples.
        Gs = self.m_f(stage, thetas, d.reshape(1, -1), xp.reshape(1, -1))
        ys = np.random.normal(loc=Gs + self.noise_loc, 
                              scale=self.noise_b_s + 
                                    self.noise_r_s * np.abs(Gs))
        # Get the corresponding one step lookahead rewards.
        expectation = 0
        for y in ys:
            expectation += self.one_step_aheah_reward(stage, xb, xp, d, y)
        return expectation / n_sample

    def optimize(self, objective, d0=None):
        """
        A function to find the optimum that maximizes objective.

        Parameters
        ----------
        objective : function
            Objective function.
        d0 : numpy.ndarray of size (n_design), optional(default=None)

        Returns
        -------
        A numpy.ndarry of size (n_design) which is the optimum.
        A float which is the maximal objective.
        """
        # We only consider 1D design space and thus use brute force here.
        assert self.n_design == 1
        max_iter = 3
        d_lim = np.copy(self.design_bounds).reshape(-1)
        for i in range(max_iter):
            n_search = 60 if i == 0 else 10
            interval = (d_lim[1] - d_lim[0]) / (n_search - 1)
            d_search = np.linspace(d_lim[0], d_lim[1], n_search)
            # Random perturbation.
            d_search += (np.random.rand(n_search) - 0.5) * interval / 2
            d_search[d_search > d_lim[1]] = d_lim[1]
            d_search[d_search < d_lim[0]] = d_lim[0]
            U_search = np.zeros(n_search)
            for j in range(n_search):
                U_search[j] = objective(np.array(d_search[j]))
            d_opt = d_search[np.argmax(U_search)]
            d_lim = np.array([max(d_opt - interval / 2, 
                                  self.design_bounds[0, 0]),
                              min(d_opt + interval / 2, 
                                  self.design_bounds[0, 1])])
        return np.array([d_opt]), max(U_search)

    def get_design(self, stage, d_hist, y_hist, 
                   xb=None, xp=None, xp_hist=None,
                   return_value=True):
        """
        A function to get the current optimal design given historical designs
        and observations.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        d_hist : numpy.ndarray of size (stage, n_design)
            The sequence of designs before stage "stage".
        y_hist : numpy.ndarray of size (stage, n_obs)
            The sequence of observations before stage "stage". d_hist and y_hist
            are used to sample parameters from the belief state.
        xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1),
             optional(default=None)
            The belief state at stage "stage".
        xp : numpy.ndarray of size (n_phys_state), optional(default=None)
            The physical state at stage "stage".
        xp_hist : numpy.ndarray of size (n_stage + 1, n_phys_state), 
                  optional(default=None)
            The historical physical states. This will be helpful if evaluating
            physical state function is expensive.
        return_vale : bool, optional(default=True)
            Whether return the approximate value function at optimum.

        Returns
        -------
        A numpy.ndarry of size (n_design) which is the optimal design.
        (optional) A float which is the approximate value function of optimal 
        design.
        """
        if xb is None:
            xb = self.get_xb(d_hist=d_hist, y_hist=y_hist)
        if xp is None:
            xp = self.get_xp(d_hist=d_hist, y_hist=y_hist)
        objective = lambda d: self.expect_on_obs(stage,
                                                 xb, xp,
                                                 d,
                                                 d_hist, y_hist,
                                                 xp_hist)
        if return_value:
            return self.optimize(objective)
        else:
            return self.optimize(objective)[0]

    def simulate_trajs(self, n_traj=None, is_explore=False, 
                       store_belief_state=True):
        """
        A function for simulating trajectories.

        Parameters
        ----------
        n_traj : int, optional(default=None)
            The trajectories simulated.
        is_explore : bool, optional(default=None)
            Whether it is exploration or exploitation.
        store_belief_state : bool, optional(default=None)
            Whether store the belief states.

        Returns
        -------
        A tuple including all information generated during the simulation, 
        including
            * thetas (parameters), numpy.ndarray of size (n_traj, n_param).
            * ds_hist (designs), numpy.ndarray of size (n_traj, 
                                                        n_stage, 
                                                        n_design).
            * ys_hist (observations), numpy.ndarray of size (n_traj,
                                                             n_stage,
                                                             n_obs) .
            * xbs_hist (belief states), could either be None or
                numpy.ndarray of size (n_traj, 
                                       n_stage,
                                       n_grid ** n_param, 
                                       n_param + 1),
                controlled by store_belief_state.
            * xps_hist (physical states), numpy.ndarray of size (n_traj,
                                                                 n_stage + 1,
                                                                 n_phys_state).
            * features_hist (features), numpy.ndarray of size (n_traj,
                                                               n_stage - 1,
                                                               n_feature).
            * rewards_hist (rewards), numpy.ndarray of size (n_traj,
                                                             n_stage + 1).
        """
        if n_traj is None: n_traj = self.n_traj
        # Generate prior samples.
        thetas = self.prior_rvs(n_traj)
        ds_hist = np.zeros((n_traj, self.n_stage, self.n_design))
        ys_hist = np.zeros((n_traj, self.n_stage, self.n_obs))
        # We store n_stage + 1 physical state while storing n_stage belief
        # state, because initial belief state is always fixed, although in this
        # code, initial physical state is also fixed, but in the future version,
        # we want to consider stochastic initial physical state. Besides, not
        # storing initial belief state can also save a lot of memory space.
        if store_belief_state:
            xbs_hist = np.zeros((n_traj, self.n_stage, *self.init_xb.shape))
        else:
            xbs_hist = None
        xps_hist = np.zeros((n_traj, self.n_stage + 1, self.n_xp))
        features_hist = np.zeros((n_traj, self.n_stage - 1, self.n_feature))
        rewards_hist = np.zeros((n_traj, self.n_stage + 1))
        n_total = self.n_stage * n_traj
        progress_points = np.rint(np.linspace(0, n_total - 1, 30))

        for i in range(n_traj):
            # Get initial belief state and physical state.
            xb = self.get_xb()
            xp = self.get_xp()
            xps_hist[i, 0] = xp
            for k in range(self.n_stage):
                # Get features.
                if k >= 1:
                    feature = self.get_features(xb, xp)
                    features_hist[i, k - 1] = feature
                # Get design.
                if is_explore:
                    # Exploration.
                    d = np.random.uniform(low=self.design_bounds[:, 0],
                                          high=self.design_bounds[:, 1])
                else:
                    # Exploitation.
                    d, _ = self.get_design(k, ds_hist[i, :i], ys_hist[i, :i],
                                           xb, xp, xps_hist[i])
                ds_hist[i, k] = d
                # Run the forward model and get observation.           
                G = self.m_f(k, 
                             thetas[i:i + 1],
                             d.reshape(1, -1),
                             xp.reshape(1, -1))
                y = np.random.normal(G + self.noise_loc,
                                     self.noise_b_s + 
                                     self.noise_r_s * np.abs(G))
                ys_hist[i, k] = y
                # Get reward.
                reward = self.get_reward(k, xb, xp, d, y)
                rewards_hist[i, k] = reward
                # Update belief state.
                xb = self.xb_f(xb, k, d, y, xp)
                if store_belief_state:
                    xbs_hist[i, k] = xb
                # Update physical state.
                xp = self.xp_f(xp.reshape(1, -1), 
                               k, 
                               d.reshape(1, -1), 
                               y.reshape(1, -1)).reshape(-1)
                xps_hist[i, k + 1] = xp
                # Progress bar.
                print('*' * (progress_points == i * self.n_stage + k).sum(),
                      end='')
            reward = self.get_reward(self.n_stage, xb, xp, None, None)
            rewards_hist[i, -1] = reward
        
        if not is_explore and n_traj > 0:
            averaged_reward = rewards_hist.sum(-1).mean()
            print("\nAveraged total reward:  {:.4}".format(averaged_reward))
        else:
            if is_explore:
                print()
        return (thetas, 
                ds_hist, ys_hist, 
                xbs_hist, xps_hist, 
                features_hist, rewards_hist)

    def approx_value_iter(self, explore_results, exploit_results):
        """
        A function to learn the feature weights of linear fit of value function
        via approximate value iteration.

        Parameters
        ----------
        explore_results : tuple
            Exploration results.
        exploit_results : tuple
            Exploitation results.
        """
        results = [np.r_[res1, res2] if res1 is not None else None 
                   for res1, res2 in zip(explore_results, exploit_results)]
        (thetas,
         ds_hist,
         ys_hist,
         xbs_hist,
         xps_hist,
         features_hist,
         rewards_hist) = results
        n_traj = len(thetas)
        Js_hist = np.zeros((n_traj, self.n_stage - 1))
        n_total = (self.n_stage - 1) * n_traj
        progress_points = np.rint(np.linspace(0, n_total - 1, 30))

        for k in range(self.n_stage - 1, 0, -1):
            for i in range(n_traj):
                if xbs_hist is None:
                    xb = None
                else:
                    xb = xbs_hist[i, k]
                xp = xps_hist[i, k]
                _, Js_hist[i, k - 1] = self.get_design(k,
                                                       ds_hist[i, :k],
                                                       ys_hist[i, :k],
                                                       xb, xp, xps_hist[i])
                print('*' * (progress_points == (k - 1) * n_traj + i).sum(),
                      end='')
            X_train = features_hist[:, k - 1, :]
            y_train = Js_hist[:, k - 1]
            reg = LinearRegression().fit(X_train, y_train)
            self.w_feature[k - 1] = reg.coef_
            self.w_feature[k - 1, 0] = reg.intercept_
        print('\n')
        return

    def soed(self, n_update=3, n_traj=1000, r_explore=0.3,
             store_belief_state=True, return_results=False):
        """
        A function to solve the sOED problem using ADP.

        Parameters
        ----------
        n_update : int, optional(default=3)
            Number of updates to find the optimal policy.  
        n_traj : int, optional(default=1000)
            Number of trajectories to sample during the training. 
        r_explore : float, optional(default=0.3)
            The ratio of exploration trajectories in all trajectories.
        store_belief_state : bool, optional(default=True)
            Whether store the belief states computed during the exploration and
            exploitation. Stored belief states will be reused in the approximate
            value iteration, thus save some forward model evaluation times. 
            However, to store the grid discritization of belief state could be 
            memory expensive, and the cost grows exponentially as the dimension 
            of parameters grows. For example, storing the belief state with 1000
            trajectories, 2 stages and 50^3 grids on a 3D parameter sapce would
            cost about 8GB. Recommend to set store_belief_state=False when 
            n_param>=3.
        return_results : bool, optional(default=False)
            Whether return exploration and exploitation results.
        """        
        assert (isinstance(n_update, int) and n_update >= 1), (
               "n_update should be a postive integer.")
        self.n_update = n_update
        assert (isinstance(n_traj, int) and n_traj >= 10), (
               "n_traj should be an integer no less than 10.")
        self.n_traj = n_traj
        assert (isinstance(r_explore, (int, float)) 
                and r_explore >= 0 
                and r_explore <= 1), (
               "r_explore should be a float between 0 and 1.")
        memory_gb = (8 * n_traj * self.n_stage * 
                     self.n_grid ** self.n_param * 
                     (self.n_param + 1) // 1000000000)
        if memory_gb > 1:
            print("Warning! The memory required to store belief states ", 
                  "exceeds 1GB. Recommend to set store_belief_state=False")
        for l in range(n_update):
            print("Update Level", self.update)
            if self.update == 0:
                self.n_explore = n_traj
                self.n_exploit = 0
            else:
                self.n_explore = int(n_traj * r_explore)
                self.n_exploit = n_traj - self.n_explore
            print("Start Exploration")
            explore_results = self.simulate_trajs(self.n_explore, 
                                                  True, 
                                                  store_belief_state)
            if self.update > 0:
                print("Start Exploitation")
            exploit_results = self.simulate_trajs(self.n_exploit, 
                                                  False, 
                                                  store_belief_state)
            print("Start Approximate Value Iteration")
            self.approx_value_iter(explore_results, exploit_results)
            self.update += 1
        print("End of SOED")
        if return_results:
            return explore_results, exploit_results

    def asses(self, n_traj=10000, 
              return_all=False,
              store_belief_state=False):
        """
        A function to asses the performance of current policy.

        Parameters
        ----------
        n_traj : int, optional(default=10000)
            Number of trajectories to sample during the assesment. 
        return_all : bool, optional(default=False)
            Return all information or not.
            If False, only return the averaged totoal reward.
            If True, return a tuple of all information generated during the 
            assesment, including 
            * averaged_reward (averaged total reward), float
            * thetas (parameters), numpy.ndarray of size (n_traj, n_param).
            * ds_hist (designs), numpy.ndarray of size (n_traj, 
                                                        n_stage, 
                                                        n_design).
            * ys_hist (observations), numpy.ndarray of size (n_traj,
                                                             n_stage,
                                                             n_obs) .
            * xbs_hist (belief states), could either be None or
                numpy.ndarray of size (n_traj, 
                                       n_stage,
                                       n_grid ** n_param, 
                                       n_param + 1),
                controlled by store_belief_state.
            * xps_hist (physical states), numpy.ndarray of size (n_traj,
                                                                 n_stage + 1,
                                                                 n_phys_state).
            * features_hist (features), numpy.ndarray of size (n_traj,
                                                               n_stage - 1,
                                                               n_feature).
            * rewards_hist (rewards), numpy.ndarray of size (n_traj,
                                                             n_stage + 1).
        store_belief_state : bool, optional(default=False)
            Whether store the belief states.

        Returns
        -------
        A float which is the averaged total reward.
        (optionally) other assesment results.
        """
        print("Start Assesment")
        asses_results = self.simulate_trajs(n_traj, 
                                            False,
                                            store_belief_state)
        (self.thetas,
         self.ds_hist,
         self.ys_hist,
         self.xbs_hist,
         self.xps_hist,
         self.features_hist,
         self.rewards_hist) = asses_results
        self.averaged_reward = self.asses_rewards_hist.sum(-1).mean()
        if not return_all:
            return self.averaged_reward
        else:
            return (self.averaged_reward, *asses_results)