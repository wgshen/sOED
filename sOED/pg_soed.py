import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .soed import SOED
from .utils import *

class PGsOED(SOED):
    """
    A class for solving sequential optimal experimental design (sOED) problems 
    using policy gradient (PG) method. This class is based on the SOED class, 
    and realize the method functions of PG upon it.
    Please refer to .

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
    actor_dimns : list, tuple or numpy.ndarray, optional(default=None)
        The dimensions of hidden layers of actor (policy) network.
    critic_dimns : list, tuple or numpy.ndarray, optional(default=None)
        The dimensions of hidden layers of critic (action value function) net.
    double_precision : bool, optional(default=False)
        Whether use double precision or single precison for the pytorch network,
        single precision is sufficiently accurate.

    Methods
    -------
    initialize()
        Initialize sOED.
    initialize_actor(), initialize_policy()
        Initialize the actor network. These two functions are equivalent.
    initialize_critic()
        Initialize the critic network.
    load_actor(), load_policy()
        Load an actor network. These two functions are equivalent.
    load_critic()
        Load an critic network. 
    get_actor(), get_policy()
        Return the actor network. These two functions are equivalent.
    get_critic()
        Return the critic network.
    form_actor_input()
        Form the inputs of actor network.
    form_critic_input()
        Form the inputs of critic network.
    get_designs()
        Get designs given sequences of historical designs and observations 
        by running the actor network.
    get_design()
        Get a single design given a sequence of historical designs and 
        observations by running the actor network.
    get_action_value()
        Get the action value (Q-value) given historical designs, observations,
        and current design.
    soed()
        Run policy gradient for give number of udpates.
    asses()
        Asses the performance of current policy.

    Future work
    -----------
    Normalize the output of neural network.
    Use MCMC or transport map to replace the grid discretization.
    Let users provide their own prior sample generator and prior PDF evaluator.
    Let users provide more complex constraints on design variables.
    Let users provide their own measurement noise function.
    Consider random initial physical state.
    Use underscore to make variables not directly accessible by users, like
    "self._n_stage", and use @property to make it indirectly accessible.
    Allow users to use gpu to accelerate the NN computation. It is not important
    in this version, because the NN is small and using cpu is fast enough.
    """
    def __init__(self, model_fun, 
                 n_stage, n_param, n_design, n_obs, 
                 prior_info, design_bounds, noise_info, 
                 reward_fun=None, phys_state_info=None,
                 n_grid=50, post_rvs_method="MCMC", random_state=None,
                 actor_dimns=None, critic_dimns=None,
                 double_precision=False):
        super().__init__(model_fun, n_stage, n_param, n_design, n_obs, 
                         prior_info, design_bounds, noise_info, 
                         reward_fun, phys_state_info,
                         n_grid, post_rvs_method, random_state)
        if random_state is None:
            random_state = np.random.randint(1e6)
        torch.manual_seed(random_state)

        assert isinstance(double_precision, bool), (
            "double_precision should either be True or False.")
        if double_precision:
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32

        # Initialize the actor (policy) network and critic network.
        self.actor_input_dimn = (self.n_stage + 
                                 (self.n_stage - 1) * (self.n_obs + 
                                                       self.n_design))
        self.critic_input_dimn = self.actor_input_dimn + self.n_design
        self.actor_dimns = actor_dimns
        self.critic_dimns = critic_dimns
        self.initialize()

        self.initialize_policy = self.initialize_actor
        self.load_policy = self.load_actor
        self.get_policy = self.get_actor

    def initialize(self):
        self.initialize_actor(self.actor_dimns)
        self.initialize_critic(self.critic_dimns)
        self.design_noise_scale = None

    def initialize_actor(self, actor_dimns=None):
        """
        Initialize the actor (policy) network.

        Parameters
        ----------
        actor_dimns : list, tuple or numpy.ndarray, optional(default=None)
            The dimensions of hidden layers of actor (policy) network.
        """
        NoneType = type(None)
        assert isinstance(actor_dimns, (list, tuple, np.ndarray, NoneType)), (
               "actor_dimns should be a list, tuple or numpy.ndarray of "
               "integers.")
        output_dimn = self.n_design
        if actor_dimns is None:
            actor_dimns = (self.actor_input_dimn * 10, 
                           self.actor_input_dimn * 10)
        self.actor_dimns = np.copy(actor_dimns)
        actor_dimns = np.append(np.append(self.actor_input_dimn, 
                                          actor_dimns), 
                                output_dimn)
        self.actor_net = Net(actor_dimns, 
                             nn.ReLU(), 
                             self.design_bounds).to(self.dtype)
        self.update = 0
        self.actor_optimizer = None
        self.actor_lr_scheduler = None

    def initialize_critic(self, critic_dimns=None):
        """
        Initialize the critic (actor-value function).

        Parameters
        ----------
        critic_dimns : list, tuple or numpy.ndarray, optional(default=None)
            The dimensions of hidden layers of critic (actor value function) 
            network.
        """
        NoneType = type(None)
        assert isinstance(critic_dimns, (list, tuple, np.ndarray, NoneType)), (
               "critic_dimns should be a list, tuple or numpy.ndarray of ",
               "integers.")
        output_dimn = 1
        if critic_dimns is None:
            critic_dimns = (self.critic_input_dimn * 10, 
                            self.critic_input_dimn * 10)
        self.critic_dimns = np.copy(critic_dimns)
        critic_dimns = np.append(np.append(self.critic_input_dimn, 
                                           critic_dimns), 
                                 output_dimn)
        self.critic_net = Net(critic_dimns, 
                              nn.ReLU(), 
                              np.array([[-np.inf, np.inf]])).to(self.dtype)
        self.update = 0
        self.critic_optimizer = None  
        self.critic_lr_scheduler = None   

    def load_actor(self, net, optimizer=None):
        """
        Load the actor network with single precision.

        Parameters
        ----------
        net : nn.Module
            A pre-trained PyTorch network with input dimension actor_input_dimn 
            and output dimension n_design.
        optimizer : algorithm of torch.optim
            An optimizer corresponding to net.
        """
        try:
            net = net.to(self.dtype)
            output = net(torch.zeros(1, self.actor_input_dimn).to(self.dtype))
            assert output.shape[1] == self.n_design, (
                   "Output dimension should be {}.".format(self.n_design))
            self.actor_net = net
        except:
            print("Actor network should has "
                  "input dimension {}.".format(self.actor_input_dimn))
        self.actor_optimizer = optimizer
        self.update = 0

    def load_critic(self, net, optimizer=None):
        """
        Load the critic network with single precision.

        Parameters
        ----------
        net : nn.Module
            A pre-trained PyTorch network with input dimension critic_input_dimn
            and output dimension 1.
        optimizer : algorithm of torch.optim
            An optimizer corresponding to net.
        """
        try:
            net = net.to(self.dtype)
            output = net(torch.zeros(1, self.critic_input_dimn).to(self.dtype))
            assert output.shape[1] == 1, (
                   "Output dimension should be 1.")
            self.critic_net = net
        except:
            print("Critic network should has "
                  "input dimension {}.".format(self.critic_input_dimn))
        self.critic_optimizer = optimizer
        self.update = 0

    def get_actor(self):
        return self.actor_net

    def get_critic(self):
        return self.critic_net

    def form_actor_input(self, stage, ds_hist, ys_hist):
        """
        A function to form the inputs of actor network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage". 

        Returns
        -------
        A torch.Tensor of size (n_traj, dimn_actor_input).
        """
        n_traj = max(len(ds_hist), len(ys_hist))
        # Inputs.
        X = np.zeros((n_traj, self.actor_input_dimn))
        # Index of experiments.
        X[:, stage] = 1
        # Historical designs.
        begin = self.n_stage
        end = begin + np.prod(ds_hist.shape[1:])
        X[:, begin:end] = ds_hist.reshape(len(ds_hist), end - begin)
        # Historical observations.
        begin = self.n_stage + (self.n_stage - 1) * self.n_design
        end = begin + np.prod(ys_hist.shape[1:])
        X[:, begin:end] = ys_hist.reshape(len(ys_hist), end - begin)
        X = torch.from_numpy(X).to(self.dtype)
        return X

    def form_critic_input(self, stage, ds_hist, ys_hist, ds):
        """
        A function to form the inputs of critic network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage". 
        ds : numpy.ndarray of size(n_traj or 1, n_design)
            Designs on which we want to get the Q value.

        Returns
        -------
        A torch.Tensor of size (n_traj, critic_input_dimn).
        """
        n_traj = max(len(ds_hist), len(ys_hist))
        X = torch.zeros(n_traj, self.critic_input_dimn).to(self.dtype)
        X[:, :self.actor_input_dimn] = self.form_actor_input(stage, 
                                                             ds_hist, 
                                                             ys_hist)
        X[:, -self.n_design:] = torch.from_numpy(ds).to(self.dtype)
        return X

    def get_designs(self, stage=0, ds_hist=None, ys_hist=None):
        """
        A function to get designs by running the policy network.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design),
                  optional(default=None)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs),
                  optional(default=None)
            n_traj sequences of observations before stage "stage". 

        Returns
        -------
        A numpy.ndarry of size (n_traj, n_design) which are designs.
        """
        if ds_hist is None:
            ds_hist = np.empty((1, 0, self.n_design))
        if ys_hist is None:
            ys_hist = np.empty((1, 0, self.n_obs))
        assert ds_hist.shape[1] == stage and ys_hist.shape[1] == stage
        X = self.form_actor_input(stage, ds_hist, ys_hist)
        designs = self.actor_net(X).detach().double().numpy()
        return designs

    def get_design(self, stage=0, d_hist=None, y_hist=None):
        """
        A function to get a single design by running the policy network.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage - 1.
        d_hist : numpy.ndarray of size (stage, n_design),
                 optional(default=None)
            A sequence of designs before stage "stage".
        y_hist : numpy.ndarray of size (stage, n_obs), 
                 optional(default=None)
            A sequence of observations before stage "stage". 

        Returns
        -------
        A numpy.ndarry of size (n_design) which is the design.
        """
        if d_hist is None:
            d_hist = np.empty((0, self.n_design))
        if y_hist is None:
            y_hist = np.empty((0, self.n_obs))
        return self.get_designs(stage,
                                d_hist.reshape(1, -1, self.n_design),
                                y_hist.reshape(1, -1, self.n_obs)).reshape(-1)

    def get_action_value(self, stage, ds_hist, ys_hist, ds):
        """
        A function to get the Q-value by running the critic network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage". 
        ds : numpy.ndarray of size(n_traj or 1, n_design)
            Designs on which we want to get the Q value.

        Returns
        -------
        A numpy.ndarry of size (n_traj) which are Q values.
        """
        assert ds_hist.shape[1] == stage and ys_hist.shape[1] == stage
        X = self.form_critic_input(stage, ds_hist, ys_hist, ds)
        values = self.critic_net(X).detach().double().numpy()
        return values.reshape(-1)

    def soed(self, n_update=100, n_traj=1000, 
             actor_optimizer=None,
             actor_lr_scheduler=None,
             n_critic_update=30,
             critic_optimizer=None,
             critic_lr_scheduler=None,
             design_noise_scale=None, design_noise_decay=0.99,
             on_policy=True):
        """
        A function to run policy gradient for given number of updates to find
        the optimal policy.

        Parameters
        ----------
        n_update : int, optional(default=3)
            Number of updates to find the optimal policy.  
        n_traj : int, optional(default=1000)
            Number of trajectories to sample during the training. 
        actor_optimizer : an algorithm of torch.optim, optional(default=None)
            The optimizer for the actor network. Example:
            torch.optim.SGD(<OBJECT INSTANCE NAME>.actor_net.parameters(),
                            lr=0.01)
        actor_lr_scheduler : a learning rate scheduler of 
                             torch.optim.lr_scheduler, optional(default=None)
            The learning rate scheduler for the actor optimizer. Example:
            torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.99)
        n_critic_update : int, optional(default=30)
            The number of updates to train the critic network within each policy
            update.
        critic_optimizer : an algorithm of torch.optim, optional(default=None)
            The optimizer for the critic network. Example:
            torch.optim.SGD(<OBJECT INSTANCE NAME>.critic_net.parameters(),
                            lr=0.01)
        critic_lr_scheduler : a learning rate scheduler of 
                              torch.optm.lr_scheduler, optional(default=None)
            The learning rate scheduler for the critic optimizer. Example:
            torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.99)
        design_noise_scale : int, list, tuple or numpy.ndarray of size 
                             (n_design), optional(default=None)
            The scale of additive exploration Gaussian noise on each dimension 
            of design variable.
        design_noise_decay : int or float, optional(default=0.99)
            The decay weight of design_noise_scale. The decay following
            design_noise_scale = design_noise_scale * design_noise_decay is 
            done after each update.
        on_policy : bool, optional(default=True)
            Whether use on-policy scheme or off-policy scheme.
            On-policy means the action value is estimated at the design that
            follows the current policy, for example,
            Q(x_k^i, d_k^i) <- g(x_k^i, d_k^i, y_k^i) +
                               Q(x_{k+1}^i, d_{k+1}^i). 
            d_{k+1}^i is the design that generated by following the policy (
            which is a noisy policy with additive noise on designs).
            Off-policy means the action value is estimated at the design that
            is not following the current policy, for example,
            Q(x_k^i, d_k^i) <- g(x_k^i, d_k^i, y_k^i) +
                               Q(x_{k+1}^i, mu(x_{k+1}^i)).
            Here the next design is mu(x_{k+1}^i), which is the clean output 
            from the actor network. Note that although mu(x_{k+1}^i) follows
            the actor network, it's not following the current true policy due
            to the lack of exploration noise. Moreover, if there is no noise
            on designs, then on-policy is equivalent to off-policy in this code.
        """
        if actor_optimizer is None:
            if self.actor_optimizer is None:
                self.actor_optimizer = optim.SGD(self.actor_net.parameters(),
                                                 lr=0.1)
        else:
            self.actor_optimizer = actor_optimizer
        if actor_lr_scheduler is None:
            if self.actor_lr_scheduler is None:
                self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.actor_optimizer, gamma=0.98)
        else:
            self.actor_lr_scheduler = actor_lr_scheduler
        if critic_optimizer is None:
            if self.critic_optimizer is None:
                self.critic_optimizer = optim.Adam(self.critic_net.parameters(),
                                                   lr=0.01)
        else:
            self.critic_optimizer = critic_optimizer
        if critic_lr_scheduler is None:
            if self.critic_lr_scheduler is None:
                self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.critic_optimizer, gamma=0.98)
        else:
            self.critic_lr_scheduler = critic_lr_scheduler
        if design_noise_scale is None:
            if self.design_noise_scale is None:
                self.design_noise_scale = (self.design_bounds[:, 1] - 
                                           self.design_bounds[:, 0]) / 20
                self.design_noise_scale[self.design_noise_scale == np.inf] = 5
        elif isinstance(design_noise_scale, (list, tuple)):
            self.design_noise_scale = np.array(design_noise_scale)
        else:
            self.design_noise_scale = design_noise_scale
        assert design_noise_decay > 0 and design_noise_decay <= 1

        for l in range(n_update):
            print('Update Level', self.update)
            self.asses(n_traj, self.design_noise_scale)

            # Form the inputs and target values of critic network, and form the 
            # inputs of the actor network.
            X_critic = torch.zeros(self.n_stage * n_traj, 
                                   self.critic_input_dimn).to(self.dtype)
            X_actor = torch.zeros(self.n_stage * n_traj,
                                  self.actor_input_dimn).to(self.dtype)
            if not on_policy:
                X_critic_off = torch.zeros_like(X_critic)
            g_critic = torch.zeros(self.n_stage * n_traj, 1).to(self.dtype)
            for k in range(self.n_stage):
                begin = k * n_traj
                end = (k + 1) * n_traj
                X = self.form_critic_input(k,
                                           self.ds_hist[:, :k],
                                           self.ys_hist[:, :k],
                                           self.ds_hist[:, k])
                X_critic[begin:end] = X
                X = self.form_actor_input(k,
                                          self.ds_hist[:, :k],
                                          self.ys_hist[:, :k])
                X_actor[begin:end] = X
                if not on_policy:
                    X = self.form_critic_input(k,
                                               self.ds_hist[:, :k],
                                               self.ys_hist[:, :k],
                                               self.dcs_hist[:, k])
                    X_critic_off[begin:end] = X
                if k == self.n_stage - 1:
                    g_critic[begin:end, 0] = torch.from_numpy(
                        self.rewards_hist[:, k:].sum(-1)).to(self.dtype)
                else:
                    if l < 20:
                        g_critic[begin:end, 0] = torch.from_numpy(
                        self.rewards_hist[:, k:].sum(-1)).to(self.dtype)
                    else:
                        if on_policy:
                            ds_next = self.ds_hist[:, k + 1]
                        else:
                            ds_next = self.dcs_hist[:, k + 1]
                        next_action_value = self.get_action_value(k + 1, 
                            self.ds_hist[:, :k + 1],
                            self.ys_hist[:, :k + 1],
                            ds_next)
                        g_critic[begin:end, 0] = torch.from_numpy(
                            self.rewards_hist[:, k] + 
                            next_action_value).to(self.dtype)
            # Train critic.
            # print(X_critic)
            # print(g_critic)
            for _ in range(n_critic_update):
                y_critic = self.critic_net(X_critic)
                loss = torch.mean((g_critic - y_critic) ** 2)
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
                # print(_, (torch.norm(y_critic - g_critic) / 
                #           torch.norm(g_critic)).item())
            self.critic_lr_scheduler.step()

            # BP to get grad_d Q(x, k)
            if on_policy:
                X_critic_back = X_critic
            else:
                X_critic_back = X_critic_off
            X_critic_back.requires_grad = True
            X_critic_back.grad = None
            output = self.critic_net(X_critic_back).sum()
            output.backward()
            critic_grad = X_critic_back.grad[:, -self.n_design:]

            # One step update on the actor network.
            # Add negative sign here because we want to do maximization.
            output = -(self.actor_net(X_actor) * critic_grad).sum(-1).mean()
            self.actor_optimizer.zero_grad()
            output.backward()
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()

            self.update += 1
            self.design_noise_scale *= design_noise_decay

    def asses(self, n_traj=10000, design_noise_scale=None,
              return_all=False, store_belief_state=False):
        """
        A function to asses the performance of current policy.

        Parameters
        ----------
        n_traj : int, optional(default=10000)
            Number of trajectories to sample during the assesment. 
        design_noise_scale : int, list, tuple or numpy.ndarray of size 
                             (n_design), optional(default=None)
            The scale of additive exploration Gaussian noise on each dimension 
            of design variable. When it is None, design_noise_scale will be
            set to 0.
        return_all : bool, optional(default=False)
            Return all information or not.
            If False, only return the averaged totoal reward.
            If True, return a tuple of all information generated during the 
            assesment, including 
            * averaged_reward (averaged total reward), float
            * thetas (parameters), numpy.ndarray of size (n_traj, n_param).
            * dcs_hist (clean designs), numpy.ndarray of size (n_traj,
                                                               n_stage,
                                                               n_design)
            * ds_hist (noisy designs), numpy.ndarray of size (n_traj, 
                                                              n_stage, 
                                                              n_design).
            * ys_hist (observations), numpy.ndarray of size (n_traj,
                                                             n_stage,
                                                             n_obs) .
            * xbs (terminal belief states), could either be None or
                numpy.ndarray of size (n_traj, 
                                       n_grid ** n_param, 
                                       n_param + 1),
                controlled by store_belief_state.
            * xps_hist (physical states), numpy.ndarray of size (n_traj,
                                                                 n_stage + 1,
                                                                 n_phys_state).
            * rewards_hist (rewards), numpy.ndarray of size (n_traj,
                                                             n_stage + 1).
        store_belief_state : bool, optional(default=False)
            Whether store the belief states.

        Returns
        -------
        A float which is the averaged total reward.
        (optionally) other assesment results.
        """
        # Generate prior samples.
        if design_noise_scale is None:
            design_noise_scale = np.zeros(self.n_design)
        elif isinstance(design_noise_scale, (int, float)):
            design_noise_scale = np.ones(self.n_design) * design_noise_scale
        elif isinstance(design_noise_scale, (list, tuple, np.ndarray)):
            assert (isinstance(design_noise_scale, (list, tuple, np.ndarray))
                    and len(design_noise_scale) == self.n_design)
        thetas = self.prior_rvs(n_traj)
        dcs_hist = np.zeros((n_traj, self.n_stage, self.n_design))
        ds_hist = np.zeros((n_traj, self.n_stage, self.n_design))
        ys_hist = np.zeros((n_traj, self.n_stage, self.n_obs))
        if store_belief_state:
            # We only store the terminal belief state.
            xbs = np.zeros((n_traj, *self.init_xb.shape))
        else:
            xbs = None
        # Store n_stage + 1 physical states.
        xps_hist = np.zeros((n_traj, self.n_stage + 1, self.n_xp))
        xps_hist[:, 0] = self.init_xp
        rewards_hist = np.zeros((n_traj, self.n_stage + 1))
        progress_points = np.rint(np.linspace(0, n_traj - 1, 30))

        for k in range(self.n_stage + 1):
            if k < self.n_stage:
                # Get clean designs.
                dcs = self.get_designs(stage=k,
                                       ds_hist=ds_hist[:, :k],
                                       ys_hist=ys_hist[:, :k])
                dcs_hist[:, k, :] = dcs
                # Add design noise for exploration.
                ds = np.random.normal(loc=dcs,
                                      scale=design_noise_scale)
                ds = np.maximum(ds, self.design_bounds[:, 0])
                ds = np.minimum(ds, self.design_bounds[:, 1])
                ds_hist[:, k, :] = ds
                # Run the forward model to get observations.
                Gs = self.m_f(k, 
                              thetas,
                              ds,
                              xps_hist[:, k, :])
                ys = np.random.normal(Gs + self.noise_loc,
                                      self.noise_b_s 
                                      + self.noise_r_s * np.abs(Gs))
                ys_hist[:, k, :] = ys
                # Get rewards.
                for i in range(n_traj):
                    rewards_hist[i, k] = self.get_reward(k, 
                                                         None, 
                                                         xps_hist[i, k],
                                                         ds[i],
                                                         ys[i])
                # Update physical state.
                xps = self.xp_f(xps_hist[:, k],
                                k,
                                ds, ys)
                xps_hist[:, k + 1] = xps
            else:
                for i in range(n_traj):
                    # Get terminal belief state.
                    xb = self.get_xb(d_hist=ds_hist[i],
                                     y_hist=ys_hist[i])
                    if store_belief_state:
                        xbs[i] = xb
                    # Get reward.
                    rewards_hist[i, k] = self.get_reward(k, 
                                                         xb, 
                                                         xps_hist[i, k],
                                                         None,
                                                         None)
                    print('*' * (progress_points == i).sum(), end='')
        averaged_reward = rewards_hist.sum(-1).mean()
        print("\nAveraged total reward:  {:.4}".format(averaged_reward))
        self.averaged_reward = averaged_reward
        self.thetas = thetas
        self.dcs_hist = dcs_hist
        self.ds_hist = ds_hist
        self.ys_hist = ys_hist
        self.xbs = xbs
        self.xps_hist = xps_hist
        self.rewards_hist = rewards_hist
        if return_all:
            return (averaged_reward, thetas, 
                    dcs_hist, ds_hist, ys_hist, 
                    xbs, xps_hist, 
                    rewards_hist)
        else:
            return averaged_reward