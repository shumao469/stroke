"""nefel.feature_extraction (compatibility layer)

The original repository started from a single notebook and a single `feature_extraction.py`.
To make the codebase easier to maintain and review, the implementation has been
split into section-based modules under `nefel.markers` plus shared helpers in
`nefel.core`.

This file keeps backward compatibility by re-exporting the most commonly used
functions.

Preferred imports (new structure)
- from nefel.core import ...
- from nefel.markers.cd31 import ...
- from nefel.markers.iba1_day1 import ...
- from nefel.markers.claudin5 import ...
- from nefel.markers.synapse import ...
- from nefel.markers.gap43 import ...
- from nefel.markers.tunel import ...
- from nefel.markers.inos_arg import ...
"""

from .core import *  # noqa

from .markers.inos_arg import *  # noqa
from .markers.iba1_day1 import *  # noqa
from .markers.claudin5 import *  # noqa
from .markers.cd31 import *  # noqa
from .markers.synapse import *  # noqa
from .markers.gap43 import *  # noqa
from .markers.tunel import *  # noqa
