"""
Guided Label Propagation (GLP) module.

This module implements the core GLP algorithm for semi-supervised community detection:
- Label probability calculation from seed nodes
- Directional propagation (in-degree and out-degree based)
- Matrix-based efficient propagation using sparse matrices
- Train/test split evaluation framework
- External validation set testing
"""

# Core propagation functions
from .propagation import (
    guided_label_propagation,
    get_propagation_info
)

# Additional modules will be added as they are implemented
# from .validation import GLPValidator
# from .evaluation import GLPEvaluator