"""
ATIO -- All Trains in One
"""
from .singleTask import MMERTrainer

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'mmer': MMERTrainer,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.model_name](args)
