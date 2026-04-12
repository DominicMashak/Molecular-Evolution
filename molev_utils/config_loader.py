"""
Domain configuration loader for MO-CMA-MAE / CMA-MAE.

A domain config YAML provides default values for all argparse arguments so
that run scripts don't need to enumerate every flag.  Command-line arguments
always override config file values.

Usage in main.py (add before the main parse):

    # --- load domain config if supplied ---
    from config_loader import inject_config_defaults
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config', type=str, default=None)
    _known, _ = _pre.parse_known_args()
    if _known.config:
        inject_config_defaults(parser, _known.config)
    # --- end config loading ---
    args = parser.parse_args()
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required for config file support. "
            "Install with: conda install -c conda-forge pyyaml"
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def inject_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """
    Load a YAML config and inject its values as argparse defaults.

    Only keys that correspond to known parser destinations are applied;
    unknown keys are silently ignored so that config files can contain
    comments and domain-specific annotations.

    The config file uses argparse dest names (underscore form):
        fitness_mode, calculator, objectives, ...

    List values (for nargs='+' args like objectives) are passed as-is.
    """
    cfg = load_yaml(config_path)

    # Build a dest → action map from all registered actions.
    dest_to_action = {action.dest: action for action in parser._actions}

    filtered: Dict[str, Any] = {}
    unknown: list = []

    for k, v in cfg.items():
        if k.startswith('_'):
            continue  # YAML comments / metadata keys
        if k in dest_to_action:
            filtered[k] = v
            # If this was a required argument, un-require it now that the
            # config provides a default value.  The post-parse validation in
            # main.py still catches the case where neither config nor CLI
            # provides the value (it will be the injected default, not None).
            action = dest_to_action[k]
            if action.required:
                action.required = False
        else:
            unknown.append(k)

    if unknown:
        import warnings
        warnings.warn(
            f"config_loader: unknown keys in {config_path}: {unknown}",
            stacklevel=2,
        )

    parser.set_defaults(**filtered)
