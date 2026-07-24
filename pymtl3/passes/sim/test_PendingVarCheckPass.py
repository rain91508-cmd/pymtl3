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


from pymtl3 import M, U


class SimplePendingModel(Component):
    """A model with a CalleeIfcCL that writes pending_x and an
    @update_once that reads pending_x — but no M<U constraint."""
    def construct(s):
        s.pending_x = None

        def _recv_method(msg):
            s.pending_x = msg
        s.recv = CalleeIfcCL(method=_recv_method, rdy=lambda: True)

        @update_once
        def up_process():
            if s.pending_x is not None:
                _ = s.pending_x
                s.pending_x = None
        s.up_process = up_process


def test_detect_missing_m_u_constraint():
    """PendingVarCheckPass should flag a missing M(recv) < U(up_process)
    constraint when recv writes pending_x and up_process reads it."""
    dut = SimplePendingModel()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "missing_m_u"
    assert "pending_x" in v.var_name
    assert "recv" in v.writer_name
    assert "up_process" in v.reader_name
