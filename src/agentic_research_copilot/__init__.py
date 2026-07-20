"""AI Research Copilot.

Clean-room research assistant inspired by open source research and agent
frameworks.
"""

from .ledger import JobLedger, RunLedger
from .pipeline import ResearchCopilot

__all__ = ["__version__", "ResearchCopilot", "RunLedger", "JobLedger"]
__version__ = "0.1.0"
