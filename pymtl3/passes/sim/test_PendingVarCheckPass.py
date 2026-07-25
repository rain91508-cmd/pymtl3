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


class _SiblingLeaf(Component):
    """A leaf component with its own pending_x and a CalleeIfcCL that
    writes it. Multiple instances of this in a parent model must NOT
    trigger cross-instance M<M/M<U violations, because each instance's
    pending_x is a distinct variable."""
    def construct(s):
        s.pending_x = []

        def _recv_method(msg):
            s.pending_x.append(msg)
        s.recv = CalleeIfcCL(method=_recv_method, rdy=lambda: True)

        @update_once
        def up_drain():
            if s.pending_x:
                s.pending_x = []
        s.up_drain = up_drain


class SiblingComponentModel(Component):
    """A parent model with two sibling child components that each declare
    their own `pending_x` variable. The pass must NOT report cross-
    instance M<M or M<U violations between the siblings, because their
    `pending_x` variables are distinct (each child's `s.pending_x` refers
    to that child's own attribute).

    Before the host-based filtering fix, the pass keyed writers/readers by
    variable name only, merging the two siblings' `pending_x` into one
    bucket and reporting spurious violations between them. This models the
    IntIQCL/MemIQCL `pending_inserts` pattern seen during IEW validation.

    Note: no M<U constraints are declared here, so within-instance M<U
    violations (a.recv->a.up_drain, b.recv->b.up_drain) are EXPECTED and
    real. The test asserts only that NO cross-instance violations exist.
    """
    def construct(s):
        s.a = _SiblingLeaf()
        s.b = _SiblingLeaf()


def test_no_cross_instance_false_positives():
    """Two sibling components with same-named pending_x variables must not
    produce cross-instance violations. Each component's s.pending_x is a
    distinct variable.

    Without the host-based filtering fix, the pass merges both siblings'
    pending_x into one bucket and reports 6 violations (2 real within-
    instance M<U + 4 cross-instance false positives: 2 M<U and 2 U<U).
    With the fix, only the 2 within-instance M<U violations remain.
    """
    dut = SiblingComponentModel()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)

    # With the fix: exactly 2 within-instance M<U violations (one per
    # child: a.recv->a.up_drain, b.recv->b.up_drain). Any cross-instance
    # pair would inflate this count.
    m_u = [v for v in violations if v.kind == "missing_m_u"]
    u_u = [v for v in violations if v.kind == "missing_u_u"]
    m_m = [v for v in violations if v.kind == "missing_m_m"]
    assert len(m_u) == 2, (
        f"Expected exactly 2 within-instance M<U violations, got "
        f"{len(m_u)}: " + "; ".join(
            f"{v.writer_name}->{v.reader_name}" for v in m_u
        )
    )
    # Cross-instance U<U (a.up_drain -> b.up_drain and vice versa) would
    # appear here without the fix. With the fix, up_drain is only compared
    # against writers in the same host, and the only same-host writer is
    # itself (skipped by the writer_obj is reader_obj check).
    assert u_u == [], (
        f"Expected 0 U<U violations (cross-instance would be false "
        f"positives), got {len(u_u)}: " + "; ".join(
            f"{v.writer_name}->{v.reader_name}" for v in u_u
        )
    )
    # Cross-instance M<M between the two 'recv' methods would appear here
    # without the fix -- but recv only writes (via .append), so no M<M
    # arises even without the fix. Still assert for completeness.
    assert m_m == [], (
        f"Expected 0 M<M violations, got {len(m_m)}: " + "; ".join(
            f"{v.writer_name}->{v.reader_name}" for v in m_m
        )
    )


class _ConstrainedLeaf(Component):
    """A leaf component with a CalleeIfcCL that writes pending_x and an
    @update_once that reads it, WITH the M<U constraint declared inside
    the child's construct. When this leaf is a child of a parent model,
    GenDAGPass propagates the M<U constraint to U<U form in the parent's
    all_constraints. The pass must recognize this propagated form."""
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


class _LeafParent(Component):
    """Parent that instantiates _ConstrainedLeaf and has an @update_once
    that calls the leaf's recv (so GenDAGPass can propagate the M<U
    constraint through this calling block)."""
    def construct(s):
        s.leaf = _ConstrainedLeaf()

        @update_once
        def up_caller():
            # Calling leaf.recv triggers M(s.leaf.recv) < U(leaf.up_process)
            # to be propagated to U(up_caller) < U(leaf.up_process) in the
            # parent's all_constraints.
            if False:
                s.leaf.recv(0)  # guarded; wiring is what matters
        s.up_caller = up_caller


def test_child_component_constraint_recognized():
    """When a child component declares M(m) < U(r) in its construct, and m
    is called from a parent @update_once block, GenDAGPass propagates the
    constraint to U(caller) < U(r) in the parent's all_constraints. The
    pass must recognize this propagated form and NOT report a false
    positive M<U violation.

    Without the propagated-form check, the pass would report a violation
    because it looks for (m, r) in all_constraints, but GenDAGPass stores
    (caller, r) instead. This models the IQCL wb_wakeup_agg ->
    up_fanout_wakeup pattern seen during IEW validation.
    """
    dut = _LeafParent()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)

    # The child's M(s.recv) < U(up_process) constraint is declared and
    # should be recognized (either directly or via propagation). No
    # violation should be reported for the recv -> up_process pair.
    recv_violations = [
        v for v in violations
        if "recv" in v.writer_name and "up_process" in v.reader_name
    ]
    assert recv_violations == [], (
        f"Expected 0 violations for recv->up_process (constraint declared "
        f"in child), got {len(recv_violations)}: " + "; ".join(
            f"{v.kind}:{v.writer_name}->{v.reader_name}"
            for v in recv_violations
        )
    )


# DefaultPassGroup is re-exported at the top level (pymtl3/__init__.py
# imports it from pymtl3.passes.PassGroups and lists it in __all__).
from pymtl3 import DefaultPassGroup


def test_default_pass_group_opt_in_check():
    """DefaultPassGroup pending-var check is opt-in via strict_check:
      strict_check=None  (default) — pass does not run
      strict_check=False             — warning mode (prints, no raise)
      strict_check=True              — strict mode (raises on violations)
    """
    # Default: pass does NOT run. SimplePendingModel has a missing M<U
    # constraint, but since the pass doesn't run, no exception.
    dut = SimplePendingModel()
    try:
        dut.apply(DefaultPassGroup())
    except Exception as e:
        pytest.fail(f"DefaultPassGroup() should not run the pass, got: {e}")

    # Warning mode: pass runs but does not raise.
    dut2 = SimplePendingModel()
    try:
        dut2.apply(DefaultPassGroup(strict_check=False))
    except Exception as e:
        pytest.fail(f"strict_check=False should not raise, got: {e}")

    # Strict mode: pass runs and raises on the missing constraint.
    dut3 = SimplePendingModel()
    with pytest.raises(Exception, match="PendingVarCheckPass"):
        dut3.apply(DefaultPassGroup(strict_check=True))


class AugAssignCounterModel(Component):
    """Multiple CalleeIfcCL methods all increment the same pending counter
    via `s.pending_count[tid] += 1` (AugAssign). Each method is an
    independent counter increment — the order between them does NOT matter
    because addition is commutative.

    Before the visit_AugAssign fix, the pass recorded the target as BOTH
    read and write (via generic_visit on the Subscript's inner Attribute
    with ctx=Load), creating false M<M cycles between all the incrementing
    methods (e.g. fu_complete/lsq_execute_resp/direct_complete all do
    s.pending_complete[tid] += 1 in WriteBackCL; ro_inst_issued_* all do
    s.pending_issued[tid] += 1).
    """
    def construct(s):
        s.pending_count = [0] * 4

        def _incr_a(tid):
            s.pending_count[tid] += 1
        s.incr_a = CalleeIfcCL(method=_incr_a, rdy=lambda: True)

        def _incr_b(tid):
            s.pending_count[tid] += 1
        s.incr_b = CalleeIfcCL(method=_incr_b, rdy=lambda: True)

        def _incr_c(tid):
            s.pending_count[tid] += 1
        s.incr_c = CalleeIfcCL(method=_incr_c, rdy=lambda: True)

        @update_once
        def up_drain():
            for tid in range(4):
                s.pending_count[tid] = 0
        s.up_drain = up_drain

        s.add_constraints(
            M(s.incr_a) < U(up_drain),
            M(s.incr_b) < U(up_drain),
            M(s.incr_c) < U(up_drain),
        )


def test_aug_assign_no_spurious_m_m_cycle():
    """`s.pending_count[tid] += 1` is an atomic read-modify-write.
    The pass must record it as WRITE only, not read+write. Recording a
    read would create false M<M cycles between methods that all increment
    the same counter.

    Without the fix, incr_a/incr_b/incr_c would all be recorded as both
    readers and writers of pending_count, producing 6 spurious M<M
    violations (a→b, a→c, b→a, b→c, c→a, c→b). With the fix, only the
    3 real M<U violations (incr_* → up_drain) are suppressed by the
    declared constraints, and 0 violations remain.
    """
    dut = AugAssignCounterModel()
    dut.elaborate()
    GenDAGPass()(dut)
    violations = PendingVarCheckPass()(dut)
    m_m = [v for v in violations if v.kind == "missing_m_m"]
    assert m_m == [], (
        f"Expected 0 M<M violations (+= is write-only, no read), got "
        f"{len(m_m)}: " + "; ".join(
            f"{v.writer_name}->{v.reader_name}" for v in m_m
        )
    )
