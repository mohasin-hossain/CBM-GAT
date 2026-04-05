import importlib
from variant_config import VARIANTS, REGISTRY

_variant_key = VARIANTS["train_model"]
_module_name = REGISTRY["train_model"][_variant_key]
_module = importlib.import_module(_module_name)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)

if __name__ == "__main__":
    main()
