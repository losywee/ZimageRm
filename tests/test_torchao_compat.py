import importlib
import sys
import types
import unittest.mock as mock

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from zinvis.regen.sdxl import peft_torchao_compat


def _reset_peft():
    sys.modules.pop("peft", None)
    sys.modules.pop("peft.import_utils", None)


def test_noop_without_peft():
    _reset_peft()
    peft_torchao_compat()  # must not raise


def test_patches_old_torchao():
    _reset_peft()
    mod = types.ModuleType("peft")
    iu = types.ModuleType("peft.import_utils")

    calls = []

    def original():
        calls.append(1)
        return True

    iu.is_torchao_available = original
    mod.import_utils = iu
    sys.modules["peft"] = mod
    sys.modules["peft.import_utils"] = iu
    with mock.patch("importlib.metadata.version",
                    return_value="0.10.0"):
        peft_torchao_compat()
    assert iu.is_torchao_available() is False
    assert not calls, "stub must replace the original"


def test_keeps_new_torchao():
    _reset_peft()
    mod = types.ModuleType("peft")
    iu = types.ModuleType("peft.import_utils")
    sentinel = lambda: True  # noqa: E731
    iu.is_torchao_available = sentinel
    mod.import_utils = iu
    sys.modules["peft"] = mod
    sys.modules["peft.import_utils"] = iu
    with mock.patch("importlib.metadata.version",
                    return_value="0.16.0"):
        peft_torchao_compat()
    assert iu.is_torchao_available is sentinel


if __name__ == "__main__":
    test_noop_without_peft()
    test_patches_old_torchao()
    test_keeps_new_torchao()
    print("TORCHAO OK")
