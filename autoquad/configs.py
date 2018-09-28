from functools import partial

from squad.utis import env_param, str2bool
from squad.dyn import mc

from .learners import TwoLayerPerceptron
from .experts import QuadLQRExpert


class base():
    #: Simulation time step.
    dt = 1/400

    #: Learner rollout length when optimizing and evaluating. Note that
    # this doesn't necessarily mean the expert has to optimize all time
    # steps.
    T = 1000

    #: DDP expert trajectory length for use during optimization.
    T_ddp = 1000

    #: Number of learner rollouts to optimize each iteration.
    num_learner = 2

    #: Number of epochs for learner to fit current dataset
    num_epochs = 10

    #: Number of DAgger iterations
    num_iters = 1000

    #: Learner evaluation is done every Nth iteration
    eval_interval = 1

    #: Number of learner rollouts during evaluation.
    num_eval = 500

    #: Limit of aggregated dataset size.
    max_aggregated = 1_000_000

    #: Use worst rollouts from evaluation as learner rollouts
    sample_method = env_param('sample_method', default='worst', cast=str)

    expert_class = QuadLQRExpert

    num_units = env_param('num_units', default=[150, 150], cast=eval)
    learner_class = partial(TwoLayerPerceptron, num_units=num_units)

class full(base):
    num_initial = 50
    num_validation = 30
    two_dimensional = env_param('two_d', default=False, cast=str2bool)
    model_class = mc.Quad2D if two_dimensional else mc.Quad
    show_tracker = env_param('show_tracker', default=False, cast=str2bool)

class quick(base):
    num_initial = 4
    num_validation = 2
    num_eval = 100
    two_dimensional = env_param('two_d', default=True, cast=str2bool)
    model_class = mc.Quad2D if two_dimensional else mc.Quad
    show_tracker = env_param('show_tracker', default=True, cast=str2bool)
