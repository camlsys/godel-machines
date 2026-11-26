# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""The host's handle on the container-side tar transfer in `node/shared/workspace_tar.py`, one implementation for both."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "node" / "shared" / "workspace_tar.py"
_SPEC = importlib.util.spec_from_file_location("rqgm_workspace_tar", _PATH)
workspace_tar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workspace_tar)
