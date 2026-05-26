"""Compatibility shim for the moved exact-trace benchmark package."""

from nlp_research_project import exact_trace_bench as _exact_trace_bench
from nlp_research_project.exact_trace_bench import *  # noqa: F403

__path__ = _exact_trace_bench.__path__
