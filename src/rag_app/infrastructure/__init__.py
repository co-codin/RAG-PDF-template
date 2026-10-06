"""Swappable backends. Each sub-package exposes a small interface (Protocol or the
LangChain base class) plus a ``build_*`` factory that picks the implementation
from Settings. Nothing above this layer imports a concrete backend directly."""
