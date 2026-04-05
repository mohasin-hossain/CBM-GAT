import importlib
from variant_config import VARIANTS, REGISTRY

_variant_key = VARIANTS["graph"]
_module_name = REGISTRY["graph"][_variant_key]
_module = importlib.import_module(_module_name)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)
