"""
src/utils/config.py

Purpose
-------
Reusable configuration helpers for the discovery ranking system.

Why this file matters
---------------------
So far, our training scripts use hardcoded Python configuration objects.
That is fine for an early baseline, but real ML repositories usually move toward
config-driven experiments so that:

- experiment settings are easier to version
- runs are easier to reproduce
- you do not need to edit source code for every experiment
- default configs and run-specific overrides can be merged cleanly

This module provides those building blocks.

What this module supports
-------------------------
1. Load config files from:
   - YAML (.yaml / .yml)
   - JSON (.json)

2. Deep-merge configs
   Useful for:
   - base config
   - experiment-specific override config

3. Validate required top-level sections

4. Save configs as JSON snapshots
   Useful for reproducible experiment artifacts

Notes
-----
- YAML loading requires PyYAML.
- JSON loading uses only the Python standard library.

Run local smoke test
--------------------
From the repo root:
    python3 -m src.utils.config
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------------------
# Optional YAML dependency
# ------------------------------------------------------------------------------

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


# ------------------------------------------------------------------------------
# Custom exception
# ------------------------------------------------------------------------------

class ConfigError(Exception):
    """
    Raised when config loading, merging, or validation fails.
    """


# ------------------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------------------

def get_repo_root() -> Path:
    """
    Return the repository root based on this file location.

    Current file:
        src/utils/config.py

    Repo root:
        ../../
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """
    Resolve a path into an absolute Path object.

    Parameters
    ----------
    path : str | Path
        Input path, absolute or relative.
    base_dir : str | Path | None
        Optional directory used to resolve relative paths. If omitted, the repo
        root is used.

    Returns
    -------
    Path
        Absolute resolved path.
    """
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    if base_dir is None:
        base_dir = get_repo_root()

    return (Path(base_dir) / path).resolve()


# ------------------------------------------------------------------------------
# Deep merge helpers
# ------------------------------------------------------------------------------

def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge two dictionaries and return a new merged dictionary.

    Merge rules
    -----------
    - If both values are dicts: merge recursively
    - Otherwise: override replaces base

    Parameters
    ----------
    base : dict[str, Any]
        Base/default configuration.
    override : dict[str, Any]
        Override/experiment-specific configuration.

    Returns
    -------
    dict[str, Any]
        New merged dictionary.

    Example
    -------
    base = {
        "model": {"max_features": 30000, "ngram_max": 2},
        "train": {"seed": 42}
    }

    override = {
        "model": {"ngram_max": 3}
    }

    result = {
        "model": {"max_features": 30000, "ngram_max": 3},
        "train": {"seed": 42}
    }
    """
    result = deepcopy(base)

    for key, override_value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(override_value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], override_value)
        else:
            result[key] = deepcopy(override_value)

    return result


# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------

def _load_json_config(path: Path) -> dict[str, Any]:
    """
    Load a JSON config file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ConfigError(f"Expected top-level JSON object in config file: {path}")

    return data


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """
    Load a YAML config file.

    Raises
    ------
    ConfigError
        If PyYAML is not installed or file content is invalid.
    """
    if yaml is None:
        raise ConfigError(
            "PyYAML is required to load YAML config files. "
            "Install it with: python3 -m pip install pyyaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Treat empty YAML as an empty dict for convenience.
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(f"Expected top-level YAML mapping in config file: {path}")

    return data


def load_config(path: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Load a config file from YAML or JSON.

    Supported file extensions
    -------------------------
    - .yaml
    - .yml
    - .json

    Parameters
    ----------
    path : str | Path
        Path to the config file.
    base_dir : str | Path | None
        Optional base directory for resolving relative paths.

    Returns
    -------
    dict[str, Any]
        Parsed config dictionary.
    """
    resolved_path = resolve_path(path, base_dir=base_dir)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    suffix = resolved_path.suffix.lower()

    if suffix == ".json":
        return _load_json_config(resolved_path)

    if suffix in {".yaml", ".yml"}:
        return _load_yaml_config(resolved_path)

    raise ConfigError(
        f"Unsupported config file extension '{suffix}' for file: {resolved_path}. "
        "Supported: .json, .yaml, .yml"
    )


def load_and_merge_configs(
    base_config_path: str | Path,
    override_config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load a base config and optionally merge in an override config.

    Parameters
    ----------
    base_config_path : str | Path
        Path to the base/default config.
    override_config_path : str | Path | None
        Optional override config path.
    base_dir : str | Path | None
        Optional base directory for resolving relative paths.

    Returns
    -------
    dict[str, Any]
        Final merged config.
    """
    base_config = load_config(base_config_path, base_dir=base_dir)

    if override_config_path is None:
        return base_config

    override_config = load_config(override_config_path, base_dir=base_dir)
    return deep_merge_dicts(base_config, override_config)


# ------------------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------------------

def require_top_level_keys(
    config: dict[str, Any],
    required_keys: list[str] | tuple[str, ...],
) -> None:
    """
    Ensure required top-level keys exist in the config.

    Parameters
    ----------
    config : dict[str, Any]
        Config dictionary to validate.
    required_keys : list[str] | tuple[str, ...]
        Required top-level keys.

    Raises
    ------
    ConfigError
        If any required keys are missing.
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ConfigError(
            "Config is missing required top-level keys: "
            + ", ".join(sorted(missing))
        )


def require_nested_keys(
    config: dict[str, Any],
    nested_key_path: list[str] | tuple[str, ...],
) -> None:
    """
    Ensure a nested config key path exists.

    Example
    -------
    require_nested_keys(config, ["model", "max_features"])
    """
    current: Any = config

    for key in nested_key_path:
        if not isinstance(current, dict) or key not in current:
            raise ConfigError(
                "Config is missing required nested key path: "
                + " -> ".join(nested_key_path)
            )
        current = current[key]


# ------------------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------------------

def _json_default_serializer(obj: Any):
    """
    Small serializer helper for json.dump.
    """
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_config_snapshot(config: dict[str, Any], output_path: str | Path) -> Path:
    """
    Save a config dictionary as a JSON snapshot.

    Parameters
    ----------
    config : dict[str, Any]
        Config to save.
    output_path : str | Path
        Output JSON path.

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=_json_default_serializer)

    return output_path


# ------------------------------------------------------------------------------
# Convenience accessors
# ------------------------------------------------------------------------------

def get_value(config: dict[str, Any], key_path: list[str] | tuple[str, ...], default: Any = None) -> Any:
    """
    Safely read a nested value from the config.

    Example
    -------
    get_value(config, ["model", "max_features"], default=30000)
    """
    current: Any = config

    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


# ------------------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local smoke test using inline dictionaries.

    This does not require any external files and is safe to run even if PyYAML
    is not installed.
    """
    print("Running config utility smoke test...\n")

    base_config = {
        "experiment": {
            "name": "baseline",
            "seed": 42,
        },
        "retrieval": {
            "model": {
                "max_features": 30000,
                "ngram_min": 1,
                "ngram_max": 2,
            },
            "train_dataset": {
                "max_impressions": 500,
                "negatives_per_positive": 4,
            },
        },
        "ranking": {
            "model": {
                "C": 1.0,
                "max_iter": 1000,
            }
        },
    }

    override_config = {
        "experiment": {
            "name": "baseline_larger_train",
        },
        "retrieval": {
            "train_dataset": {
                "max_impressions": 1000,
            }
        },
        "ranking": {
            "model": {
                "C": 0.5,
            }
        },
    }

    merged_config = deep_merge_dicts(base_config, override_config)

    print("Merged config:")
    print(json.dumps(merged_config, indent=2))

    print("\nValidating required keys...")
    require_top_level_keys(merged_config, ["experiment", "retrieval", "ranking"])
    require_nested_keys(merged_config, ["retrieval", "model", "max_features"])
    require_nested_keys(merged_config, ["ranking", "model", "C"])
    print("Validation passed.")

    print("\nNested access examples:")
    print(
        "retrieval.model.max_features =",
        get_value(merged_config, ["retrieval", "model", "max_features"]),
    )
    print(
        "ranking.model.C =",
        get_value(merged_config, ["ranking", "model", "C"]),
    )
    print(
        "missing.path.with.default =",
        get_value(merged_config, ["missing", "path"], default="fallback"),
    )

    snapshot_path = get_repo_root() / "artifacts" / "logs" / "config_smoke_test.json"
    saved_path = save_config_snapshot(merged_config, snapshot_path)
    print(f"\nSaved smoke-test config snapshot to: {saved_path}")