"""specid -- spectral identifiability of olivine-pyroxene mixtures: forward model, continuous library, identifiability engine,
learned retrieval with distribution-free regions, optimal band design and the near-Earth-asteroid application.

The modules are flat (they import each other by their bare names), so the package directory must be on sys.path; importing
`specid` does that:  `import specid; from specid import pipeline`."""
import sys as _sys
from pathlib import Path as _Path
__version__ = '1.0.0'
_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
