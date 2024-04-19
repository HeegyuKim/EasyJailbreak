from ..models import BlackBoxModelBase

class BaseDefense(BlackBoxModelBase):
    def __init__(self, model):
        self.model = model

    def generate(self, messages, clear_old_history=True, **kwargs):
        return self.model.generate(messages, clear_old_history=clear_old_history, **kwargs)

    def batch_generate(self, conversations, **kwargs):
        return self.model.batch_generate(conversations, **kwargs)