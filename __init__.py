# This file makes the directory a Python package.
# It tells ComfyUI that it can look inside this directory for custom nodes.

# Import nodes from the main script to make them available
from .craftsman_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[ComfyUI-CraftsManWrapper] Initialized custom node package.")
