"""Tests for scripts/pyproject_deps.py — the pyproject → pip-requirements extractor the Docker images install from."""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("pyproject_deps", os.path.join(_ROOT, "scripts", "pyproject_deps.py"))
pyproject_deps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pyproject_deps)

# The extras Dockerfile.api / Dockerfile.celery pass. Keep in sync with both.
IMAGE_EXTRAS = ["ml", "modal", "tracking", "dev"]


def _write(tmp_path, body):
    path = tmp_path / "pyproject.toml"
    path.write_text(body)
    return str(path)


FAKE = """
[project]
name = "My_Proj"
dependencies = ["fastapi==0.104.1", "numpy>=1.26"]

[project.optional-dependencies]
ml = ["torch>=2.1.0", "numpy>=1.26"]
dev = ["pytest==7.4.3"]
gpu = ["my-proj[ml]", "boltz @ git+https://example.com/boltz@abc"]

[tool.other]
dependencies = ["poison==9.9.9"]
"""


def test_core_then_extras_in_order_deduplicated(tmp_path):
    reqs = pyproject_deps.requirements(_write(tmp_path, FAKE), ["ml", "dev"])
    assert reqs == ["fastapi==0.104.1", "numpy>=1.26", "torch>=2.1.0", "pytest==7.4.3"]


def test_core_only(tmp_path):
    assert pyproject_deps.requirements(_write(tmp_path, FAKE), []) == ["fastapi==0.104.1", "numpy>=1.26"]


def test_self_reference_rejected_with_pep503_normalisation(tmp_path):
    # "My_Proj" vs "my-proj[ml]" — must still be recognised as the project itself.
    with pytest.raises(ValueError, match="references the project itself"):
        pyproject_deps.requirements(_write(tmp_path, FAKE), ["gpu"])


def test_unknown_extra_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown extra"):
        pyproject_deps.requirements(_write(tmp_path, FAKE), ["nope"])


def test_main_exit_codes(tmp_path, capsys):
    path = _write(tmp_path, FAKE)
    assert pyproject_deps.main(["x", path, "ml"]) == 0
    assert "torch>=2.1.0" in capsys.readouterr().out
    assert pyproject_deps.main(["x", path, "gpu"]) == 1
    assert pyproject_deps.main(["x"]) == 2


def test_real_pyproject_image_extras_are_leaf_and_complete():
    """The image install set resolves from the real pyproject and carries what the images need."""
    reqs = pyproject_deps.requirements(os.path.join(_ROOT, "pyproject.toml"), IMAGE_EXTRAS)
    names = {pyproject_deps._normalize(pyproject_deps._NAME.match(r).group(1)) for r in reqs}
    # Runtime entry points of the two images + the heavy ML stack the worker imports.
    for needed in ("fastapi", "uvicorn", "celery", "flower", "sqlalchemy", "torch", "transformers", "numpy"):
        assert needed in names
    # Benchmark-only and GPU deps deliberately stay out of the Docker images.
    assert not names & {"pandas", "scipy", "pyarrow", "boltz"}


def _modal_image_extras():
    """MODAL_IMAGE_EXTRAS from modal_app.py, read via ast so the test doesn't need `modal` installed."""
    import ast

    with open(os.path.join(_ROOT, "modal_app.py")) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODAL_IMAGE_EXTRAS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("MODAL_IMAGE_EXTRAS not found in modal_app.py")


def _dockerfile_extras(name):
    """The extras a Dockerfile passes to pyproject_deps.py."""
    import re

    with open(os.path.join(_ROOT, name)) as fh:
        match = re.search(r"pyproject_deps\.py\s+\S*pyproject\.toml\s+([\w ]+?)\s*>", fh.read())
    assert match, f"{name} no longer installs via pyproject_deps.py"
    return match.group(1).split()


@pytest.mark.parametrize("dockerfile", ["Dockerfile.api", "Dockerfile.celery"])
def test_dockerfiles_use_image_extras(dockerfile):
    assert _dockerfile_extras(dockerfile) == IMAGE_EXTRAS


def test_modal_image_extras_match_and_are_valid_leaf_extras():
    """Modal silently skips unknown extras and passes self-references to pip — so check here."""
    extras = _modal_image_extras()
    assert extras == IMAGE_EXTRAS
    # Raises on an unknown name (typo) or a `propredict[...]` self-reference.
    pyproject_deps.requirements(os.path.join(_ROOT, "pyproject.toml"), extras)
