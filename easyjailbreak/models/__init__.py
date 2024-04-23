from .model_base import ModelBase, WhiteBoxModelBase, BlackBoxModelBase
from .huggingface_model import HuggingfaceModel, from_pretrained
from .openai_model import OpenaiModel
from .wenxinyiyan_model import WenxinyiyanModel
from .flax_api import FlaxAPI


try:
    import jax, flax
    INSTALL_FLAX = True
except ImportError:
    print("JAX and Flax not installed, skipping import of FlaxHuggingfaceModel")
    INSTALL_FLAX = False
    
if INSTALL_FLAX:
    from .flax_huggingface_model import FlaxHuggingfaceModel
    __all__ = ['ModelBase', 'WhiteBoxModelBase', 'BlackBoxModelBase', 'HuggingfaceModel', 'from_pretrained', 'OpenaiModel', 'WenxinyiyanModel', 'FlaxHuggingfaceModel', 'FlaxAPI']
else:
    __all__ = ['ModelBase', 'WhiteBoxModelBase', 'BlackBoxModelBase', 'HuggingfaceModel', 'from_pretrained', 'OpenaiModel', 'WenxinyiyanModel']