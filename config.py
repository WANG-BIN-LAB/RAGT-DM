class Config:
    def __init__(self):
        # -------------------------- Dataset Settings -------------------------- #
        self.dataset = type('', (), {})()
        self.dataset.name = 'abide'
        self.dataset.batch_size = 16                # Training batch size
        self.dataset.test_batch_size = 16            # Test batch size
        self.dataset.path = 'abide.npy'  # General path (removed personal info)
        self.dataset.stratified = True               # Stratified sampling for CV
        self.dataset.drop_last = True                # Drop last incomplete batch

        # -------------------------- Preprocessing -------------------------- #
        self.preprocess = type('', (), {})()
        self.preprocess.name = 'default'
        self.preprocess.continus = True              # Continuous data mode

        # -------------------------- Model Settings -------------------------- #
        self.model = type('', (), {})()
        self.model.name = 'BrainNetworkTransformer'
        self.model.pos_encoding = True
        self.model.pos_embed_dim = 200
        self.model.sizes = [200, 100, 50]
        self.model.orthogonal = True
        self.model.freeze_center = False
        self.model.project_assignment = True
        # Transformer backbone params
        self.model.num_layers = 1
        self.model.nhead = 4
        self.model.global_topk_ratio = 0.3
        self.model.local_neighbor_num = 9

        # -------------------------- Training Settings -------------------------- #
        self.training = type('', (), {})()
        self.training.epochs = 100                   # Total training epochs
        self.total_steps = 10000                     # Total training steps
        self.save_learnable_graph = False            # Disable save learnable graph
        self.n_folds = 10                            # 10-fold cross validation
        self.seed = 42                               # Global random seed

        # -------------------------- Optimizer Settings -------------------------- #
        self.optimizer = type('', (), {})()
        self.optimizer.name = 'Adam'
        self.optimizer.lr = 5.0e-5                   # Fixed learning rate
        self.optimizer.weight_decay = 1.0e-4         # L2 regularization

        # -------------------------- Other Settings -------------------------- #
        self.project = 'brain_network'