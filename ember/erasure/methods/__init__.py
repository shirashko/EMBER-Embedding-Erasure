"""Per-method plug-ins. Importing this package registers every method.

Each module calls :func:`base.register` at import time, so by the time this
``__init__`` is loaded the :data:`base.REGISTRY` dict is populated.
"""
from ember.erasure.methods import base
from ember.erasure.methods import snmf
from ember.erasure.methods import rmu
from ember.erasure.methods import crisp
from ember.erasure.methods import ember
from ember.erasure.methods import pisces

__all__ = ["base"]
