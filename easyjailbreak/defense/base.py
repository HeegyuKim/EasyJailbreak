from typing import Any
from ..models import BlackBoxModelBase

class BaseDefense(BlackBoxModelBase):
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)
    
    def generate(self, messages, clear_old_history=True, **kwargs):
        return self.model.generate(messages, clear_old_history=clear_old_history, **kwargs)

    def batch_generate(self, conversations, **kwargs):
        return self.model.batch_generate(conversations, **kwargs)