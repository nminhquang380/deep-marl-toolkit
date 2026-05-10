class MultiUAVQMixConfig:
    """QMIX hyperparameters for multi-UAV trajectory optimization."""

    model_type: str = 'qmix'

    def __init__(
        self,
        fc_hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        gamma: float = 0.99,
        egreedy_exploration: float = 1.0,
        min_exploration: float = 0.01,
        target_update_tau: float = 0.05,
        target_update_interval: int = 100,
        learning_rate: float = 5e-4,
        min_learning_rate: float = 1e-6,
        clip_grad_norm: float = 10.0,
        hypernet_layers: int = 2,
        hypernet_embed_dim: int = 64,
        mixing_embed_dim: int = 32,
        learner_update_freq: int = 1,
        double_q: bool = True,
        algo_name: str = 'qmix',
        # Environment reward shaping
        phi: float = 0.5,
        rho_col: float = 1.0,
        r_fail: float = 5.0,
        # Checkpointing
        save_interval: int = 200,
    ) -> None:
        self.fc_hidden_dim = fc_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.hypernet_layers = hypernet_layers
        self.hypernet_embed_dim = hypernet_embed_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.gamma = gamma
        self.egreedy_exploration = egreedy_exploration
        self.min_exploration = min_exploration
        self.target_update_tau = target_update_tau
        self.target_update_interval = target_update_interval
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.learner_update_freq = learner_update_freq
        self.double_q = double_q
        self.algo_name = algo_name
        self.phi = phi
        self.rho_col = rho_col
        self.r_fail = r_fail
        self.save_interval = save_interval
