# PendingVarCheckPass.py
"""PendingVarCheckPass — static check for missing M<U/U<U constraints
on shared Python variables (pending_*, next_*, state_*).

PyMTL3's GenDAGPass only tracks Signal reads/writes. Python list/dict
variables (s.pending_*, s.next_*, s.state_*) are invisible to the
constraint DAG. This pass performs custom AST analysis to detect
missing ordering constraints between CalleeIfcCL methods and
@update_once blocks that share these variables.

Run after GenDAGPass, before SimpleSchedulePass/DynamicSchedulePass.
"""
from pymtl3.passes.BasePass import BasePass, PassMetadata


class PendingVarCheckPass(BasePass):
    """Detect missing constraints on shared Python variables.

    Returns a list of Violation objects. In warning mode (default),
    prints warnings to stderr. In strict mode, raises an exception.
    """

    def __init__(self, strict=False):
        super().__init__()
        self.strict = strict

    def __call__(self, top):
        violations = []
        # TODO: implement analysis
        return violations
