import inspect
import pkgutil
import importlib
from pathlib import Path
import data_ingestion.connectors as connectors_pkg


def iter_connector_classes():
    pkg_path = Path(connectors_pkg.__file__).parent
    for modinfo in pkgutil.iter_modules([str(pkg_path)]):
        if modinfo.name.startswith("_"):
            continue
        module = importlib.import_module(f"{connectors_pkg.__name__}.{modinfo.name}")
        for name, obj in vars(module).items():
            if isinstance(obj, type) and getattr(obj, "FETCH_SIGNATURE_OK", False):
                yield obj


def test_connectors_have_standard_signature():
    for cls in iter_connector_classes():
        sig = inspect.signature(cls.fetch_data)
        params = list(sig.parameters.values())
        expected = ["symbols", "timeframe", "start_date", "end_date"]
        assert [p.name for p in params] == expected, (
            f"{cls.__name__}.fetch_data must accept {expected}"
        )
