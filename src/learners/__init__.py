from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .vaos_learner import VAOSLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["vaos_learner"] = VAOSLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
