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


class ConstrainedPendingModel(Component):
    """Same as SimplePendingModel but WITH M<U constraint."""
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

        s.add_constraints(M(s.recv) < U(up_process))


def test_no_violation_when_constraint_exists():
    """When M(recv) < U(up_process) is declared, no violation."""
    dut = ConstrainedPendingModel()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)
    assert len(violations) == 0, f"Expected 0 violations, got {violations}"


class StateWriteViolationModel(Component):
    """CalleeIfcCL writes state_* directly — CL discipline violation."""
    def construct(s):
        s.state_x = [0]

        def _recv_method(msg):
            s.state_x[0] = msg  # Direct state_* write by CalleeIfcCL
        s.recv = CalleeIfcCL(method=_recv_method, rdy=lambda: True)

        @update_once
        def up_process():
            _ = s.state_x[0]
        s.up_process = up_process


def test_detect_cl_discipline_violation():
    """CalleeIfcCL writing state_* should be flagged as CL discipline
    violation."""
    dut = StateWriteViolationModel()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)
    assert len(violations) >= 1
    v = violations[0]
    assert v.kind == "cl_discipline"
    assert "state_x" in v.var_name


class SharedAppendListModel(Component):
    """Two CalleeIfcCL methods both .append() to the same pending list
    (write-only), and an @update_once reads+clears it. With M<U
    constraints declared, there should be NO violations.

    This models the FUPool pending_fast_path_completes pattern where
    fu_operand_int/fu_operand_float/... all append and up_fire_completions
    drains. Before the visit_Call fix, .append() was recorded as both
    write AND read (via generic_visit on node.func), creating spurious
    M<M violations between the appending methods.
    """
    def construct(s):
        s.pending_list = []

        def _writer_a(msg):
            s.pending_list.append(msg)
        s.writer_a = CalleeIfcCL(method=_writer_a, rdy=lambda: True)

        def _writer_b(msg):
            s.pending_list.append(msg)
        s.writer_b = CalleeIfcCL(method=_writer_b, rdy=lambda: True)

        @update_once
        def up_drain():
            for i in range(len(s.pending_list)):
                _ = s.pending_list[i]
            s.pending_list = []
        s.up_drain = up_drain

        s.add_constraints(
            M(s.writer_a) < U(up_drain),
            M(s.writer_b) < U(up_drain),
        )


def test_append_is_write_only_no_spurious_m_m():
    """.append() on a shared pending_* list must be recorded as write-only,
    not write+read. Without the fix, generic_visit on the Call's func
    triggers visit_Attribute on the receiver, recording a spurious READ
    that creates M<M violations between the appending methods."""
    dut = SharedAppendListModel()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)
    assert len(violations) == 0, (
        f"Expected 0 violations (M<U constraints exist, .append() is "
        f"write-only), got {len(violations)}: "
        + "; ".join(
            f"{v.kind}:{v.writer_name}->{v.reader_name}" for v in violations
        )
    )


# DefaultPassGroup is re-exported at the top level (pymtl3/__init__.py
# imports it from pymtl3.passes.PassGroups and lists it in __all__).
from pymtl3 import DefaultPassGroup


def test_default_pass_group_includes_check():
    """DefaultPassGroup should run PendingVarCheckPass automatically
    in warning mode (no raise)."""
    dut = SimplePendingModel()  # has missing M<U constraint
    try:
        dut.apply(DefaultPassGroup())
    except Exception as e:
        pytest.fail(f"DefaultPassGroup should not raise in warning mode, got: {e}")

    # In strict mode, it should raise
    dut2 = SimplePendingModel()
    with pytest.raises(Exception, match="PendingVarCheckPass"):
        dut2.apply(DefaultPassGroup(strict_check=True))
