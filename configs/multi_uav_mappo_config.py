class MultiUAVMAPPOConfig:
    """MAPPO hyperparameters for multi-UAV trajectory optimization."""

    model_type: str = 'mappo'

    def __init__(
        self,
        # Network
        hidden_size: int = 128,
        rnn_layers: int = 1,
        use_recurrent_policy: bool = False,
        use_naive_recurrent_policy: bool = False,
        use_orthogonal: bool = True,
        use_feature_normalization: bool = True,
        activation: str = 'relu',
        gain: float = 0.01,
        data_chunk_length: int = 10,
        # PPO
        ppo_epoch: int = 10,
        num_mini_batch: int = 4,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 10.0,
        use_max_grad_norm: bool = True,
        use_clipped_value_loss: bool = True,
        use_huber_loss: bool = True,
        huber_delta: float = 10.0,
        use_value_active_masks: bool = True,
        use_policy_active_masks: bool = True,
        # GAE / returns
        use_gae: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_proper_time_limits: bool = False,
        # Value normalisation
        use_popart: bool = False,
        use_valuenorm: bool = True,
        # Optimiser
        actor_lr: float = 5e-4,
        critic_lr: float = 5e-4,
        opti_eps: float = 1e-5,
        weight_decay: float = 0.0,
        # Logging
        algo_name: str = 'mappo',
        algorithm_name: str = 'mappo',
        phi: float = 0.5,
        rho_col: float = 1.0,
        r_fail: float = 5.0,
        save_interval: int = 50,
    ) -> None:
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.use_recurrent_policy = use_recurrent_policy
        self.use_naive_recurrent_policy = use_naive_recurrent_policy
        self.use_orthogonal = use_orthogonal
        self.use_feature_normalization = use_feature_normalization
        self.activation = activation
        self.gain = gain
        self.data_chunk_length = data_chunk_length
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_max_grad_norm = use_max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.use_value_active_masks = use_value_active_masks
        self.use_policy_active_masks = use_policy_active_masks
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        self.use_popart = use_popart
        self.use_valuenorm = use_valuenorm
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.opti_eps = opti_eps
        self.weight_decay = weight_decay
        self.algo_name = algo_name
        self.algorithm_name = algorithm_name
        self.phi = phi
        self.rho_col = rho_col
        self.r_fail = r_fail
        self.save_interval = save_interval
