# ============================================================
# SPIKEINTERFACE AUTOMATED CURATION
# ============================================================

# Core objects
import spikeinterface as si

from spikeinterface.core import (
    create_sorting_analyzer,
    load_sorting_analyzer,
    compute_sparsity,
)

# ------------------------------------------------------------
# POSTPROCESSING
# ------------------------------------------------------------

from spikeinterface.postprocessing import (
    compute_spike_amplitudes,
    compute_unit_locations,
    compute_spike_locations,
    compute_template_similarity,
)

# ------------------------------------------------------------
# QUALITY AND TEMPLATE METRICS
# ------------------------------------------------------------

from spikeinterface.metrics.quality import (
    compute_quality_metrics,
    get_quality_metric_list,
    get_default_quality_metrics_params,
)

from spikeinterface.metrics.template import (
    compute_template_metrics,
    get_template_metric_list,
)

# ------------------------------------------------------------
# AUTOMATIC CURATION
# ------------------------------------------------------------

from spikeinterface.curation import (
    model_based_label_units,
    get_required_metrics_from_model,
    bombcell_label_units,
    threshold_metrics,
)

# ------------------------------------------------------------
# DUPLICATE DETECTION
# ------------------------------------------------------------

from spikeinterface.curation import (
    find_redundant_units,
    remove_redundant_units,
    remove_duplicated_spikes,
)

# ------------------------------------------------------------
# AUTOMATIC MERGING
# ------------------------------------------------------------

from spikeinterface.curation import (
    compute_merge_unit_groups,
    CurationSorting,
)

# ------------------------------------------------------------
# VISUAL REVIEW AND EXPORT
# ------------------------------------------------------------

from spikeinterface.widgets import (
    plot_sorting_summary,
    plot_unit_templates,
)

from spikeinterface.exporters import (
    export_report,
    export_to_phy,
)

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd