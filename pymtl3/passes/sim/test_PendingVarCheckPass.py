# test_PendingVarCheckPass.py
"""Tests for PendingVarCheckPass — static check for missing constraints
on shared Python variables (pending_*, next_*, state_*)."""
import pytest
from pymtl3 import Component, CalleeIfcCL, CallerIfcCL, update_once, update_ff
from pymtl3.passes.sim.GenDAGPass import GenDAGPass
from pymtl3.passes.sim.PendingVarCheckPass import PendingVarCheckPass


class EmptyModel(Component):
    def construct(s):
        pass


def test_pass_runs_on_empty_model():
    """PendingVarCheckPass should run without error on a model with
    no pending_* variables."""
    dut = EmptyModel()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)
    assert violations == []
