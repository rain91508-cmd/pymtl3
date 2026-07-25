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
import ast
import inspect
import sys
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from pymtl3.passes.BasePass import BasePass


# Variables to track: pending_*, next_*, state_* (with optional _ prefix)
_VAR_PATTERN = re.compile(r'^_?(pending_|next_|state_)')


@dataclass
class VarAccess:
    """A single access to a tracked variable."""
    var_name: str          # e.g. "pending_fast_path_completes"
    access_type: str       # "read" or "write"
    method_or_block: str   # human-readable name of the method/block


@dataclass
class Violation:
    """A detected constraint violation."""
    kind: str              # "missing_m_u", "missing_u_u", "missing_m_m",
                           # "cl_discipline", "constraint_cycle"
    var_name: str          # the shared variable
    writer_name: str       # method/block that writes
    reader_name: str       # method/block that reads
    message: str           # human-readable description
    suggestion: str = ""   # suggested fix


class _PendingVarVisitor(ast.NodeVisitor):
    """AST visitor that tracks reads/writes of pending_*/next_*/state_*
    variables accessed via s.* attributes.

    Tracks:
    - Direct read: s.pending_x, for x in s.pending_x, len(s.pending_x)
    - Direct write: s.pending_x = ..., s.pending_x.append(...),
      s.pending_x.clear(), s.pending_x = []
    - Indirect field write: b.field = val (where b was assigned from
      a tracked variable, e.g. b = s.state_complete[fu])
    """

    def __init__(self):
        self.accesses = []        # list of VarAccess
        self._alias_map = {}      # local_var -> tracked_var_name

    def _is_tracked_name(self, name_parts):
        """Check if a dotted name like ['s', 'pending_x'] refers to a
        tracked variable. Returns the variable name or None."""
        if len(name_parts) < 2:
            return None
        if name_parts[0] != 's':
            return None
        if _VAR_PATTERN.match(name_parts[1]):
            return name_parts[1]
        return None

    def _get_full_name(self, node):
        """Extract dotted name from an AST node. Returns list of parts
        or None if not a simple dotted name.

        Peels off Subscript wrappers (e.g. s.state_x[0] -> s.state_x) so
        that subscripted reads/writes of tracked variables are recognized
        as accesses to the underlying tracked name (state_x). Without
        this, `s.state_x[0] = msg` would be missed entirely -- the
        Subscript node is not an Attribute, so the Attribute-extraction
        loop below would not extract 's.state_x', and _handle_target /
        visit_Subscript would record nothing.
        """
        parts = []
        while isinstance(node, ast.Subscript):
            node = node.value
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        else:
            return None
        parts.reverse()
        return parts

    def visit_Assign(self, node):
        # Visit value first (reads on RHS)
        self.visit(node.value)
        # Then visit targets (writes)
        for target in node.targets:
            self._handle_target(target)

    def visit_AugAssign(self, node):
        # e.g. s.pending_x[tid] += 1 (atomic read-modify-write)
        # Record ONLY a write on the target. The read is part of the
        # atomic operation and does NOT create a data dependency on other
        # methods' writes. Recording a read would create false M<M cycles
        # between methods that all increment the same counter (e.g.
        # fu_complete, lsq_execute_resp, direct_complete all do
        # s.pending_complete[tid] += 1; ro_inst_issued_* all do
        # s.pending_issued[tid] += 1).
        #
        # We use _handle_target (which records the write without visiting
        # children) instead of self.visit(node.target) (which would dispatch
        # to visit_Subscript/visit_Attribute and call generic_visit,
        # recording a spurious READ via the inner Attribute's ctx=Load).
        self._handle_target(node.target)
        self.visit(node.value)
        # Visit Subscript slice for tracked variable reads in the index
        # (e.g. s.pending_x[s.pending_y] += 1 — pending_y is read).
        if isinstance(node.target, ast.Subscript):
            self.visit(node.target.slice)

    def visit_For(self, node):
        # for x in s.pending_x:  -> read of s.pending_x
        # Also track alias: x = element from s.pending_x
        iter_name = self._get_full_name(node.iter)
        if iter_name:
            tracked = self._is_tracked_name(iter_name)
            if tracked:
                self.accesses.append(VarAccess(
                    var_name=tracked, access_type="read",
                    method_or_block=""
                ))
                # Track loop variable as alias
                if isinstance(node.target, ast.Name):
                    self._alias_map[node.target.id] = tracked
        self.generic_visit(node)

    def visit_Call(self, node):
        # Detect s.pending_x.append(...), s.pending_x.clear(), etc.
        func_name = self._get_full_name(node.func)
        is_mutating = False
        if func_name and len(func_name) >= 3:
            if func_name[0] == 's' and _VAR_PATTERN.match(func_name[1]):
                # s.pending_x.append(...) -> write
                mutating_methods = {
                    'append', 'extend', 'clear', 'pop', 'insert',
                    'remove', 'popitem', 'setdefault', 'update',
                }
                if func_name[2] in mutating_methods:
                    self.accesses.append(VarAccess(
                        var_name=func_name[1], access_type="write",
                        method_or_block=""
                    ))
                    is_mutating = True
        if is_mutating:
            # Only visit args/kwargs, NOT node.func. Visiting node.func
            # would trigger visit_Attribute on the receiver (s.pending_x),
            # recording a spurious READ -- but accessing the bound .append
            # method is part of the write, not a separate read of the list
            # contents. This false read caused spurious M<M violations
            # between CalleeIfcCL methods that all .append() to the same
            # pending list (e.g. fu_operand_int/fu_operand_float sharing
            # pending_fast_path_completes in FUPool).
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node):
        # Detect s.pending_x read/write
        name_parts = self._get_full_name(node)
        if name_parts:
            tracked = self._is_tracked_name(name_parts)
            if tracked:
                if isinstance(node.ctx, ast.Load):
                    self.accesses.append(VarAccess(
                        var_name=tracked, access_type="read",
                        method_or_block=""
                    ))
                elif isinstance(node.ctx, ast.Store):
                    self.accesses.append(VarAccess(
                        var_name=tracked, access_type="write",
                        method_or_block=""
                    ))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        # s.pending_x[0] read, or s.state_complete[fu] read
        name_parts = self._get_full_name(node)
        if name_parts:
            tracked = self._is_tracked_name(name_parts)
            if tracked:
                if isinstance(node.ctx, ast.Load):
                    self.accesses.append(VarAccess(
                        var_name=tracked, access_type="read",
                        method_or_block=""
                    ))
                    # Track alias: b = s.state_complete[fu]
                    # (handled in visit_Assign via _handle_target)
                elif isinstance(node.ctx, ast.Store):
                    self.accesses.append(VarAccess(
                        var_name=tracked, access_type="write",
                        method_or_block=""
                    ))
        self.generic_visit(node)

    def _handle_target(self, target):
        """Handle assignment targets, including alias tracking."""
        if isinstance(target, ast.Name):
            # Plain assignment: x = ... (may be alias from RHS)
            pass  # alias tracking handled in visit_Assign
        elif isinstance(target, ast.Attribute):
            # Could be s.pending_x = ... or b.field = ...
            name_parts = self._get_full_name(target)
            if name_parts:
                tracked = self._is_tracked_name(name_parts)
                if tracked:
                    self.accesses.append(VarAccess(
                        var_name=tracked, access_type="write",
                        method_or_block=""
                    ))
                elif len(name_parts) == 2 and name_parts[0] in self._alias_map:
                    # Indirect field write: b.field = val where b is aliased
                    # to a tracked variable
                    tracked_var = self._alias_map[name_parts[0]]
                    self.accesses.append(VarAccess(
                        var_name=tracked_var, access_type="write",
                        method_or_block=""
                    ))
        elif isinstance(target, ast.Subscript):
            # s.pending_x[i] = ... or b[i] = ...
            name_parts = self._get_full_name(target)
            if name_parts:
                tracked = self._is_tracked_name(name_parts)
                if tracked:
                    self.accesses.append(VarAccess(
                        var_name=tracked, access_type="write",
                        method_or_block=""
                    ))
                elif name_parts[0] in self._alias_map:
                    tracked_var = self._alias_map[name_parts[0]]
                    self.accesses.append(VarAccess(
                        var_name=tracked_var, access_type="write",
                        method_or_block=""
                    ))

    def analyze(self, source_code, block_name=""):
        """Parse and analyze source code. Returns list of VarAccess.

        The source is dedented before parsing to handle nested functions
        (inspect.getsource returns indented source for closures).
        """
        try:
            dedented = textwrap.dedent(source_code)
            tree = ast.parse(dedented)
        except SyntaxError:
            return []
        try:
            self.visit(tree)
        except Exception as e:
            # Defensive: real-world models may contain AST constructs the
            # visitor doesn't handle. Skip this block rather than crash the
            # whole pass.
            print(f"[PendingVarCheck] WARNING: AST visitor failed on "
                  f"{block_name}: {e}", file=sys.stderr)
            return []
        for acc in self.accesses:
            if not acc.method_or_block:
                acc.method_or_block = block_name
        return self.accesses


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

        # 1. Collect all update_once blocks and their source.
        # Use inspect.getsource directly on the function object for a
        # uniform approach that works regardless of which component
        # class caches the metadata. The source is dedented inside
        # _PendingVarVisitor.analyze.
        all_uponce = top.get_all_update_once()
        blk_info = {}
        for blk in all_uponce:
            try:
                name = blk.__name__
            except Exception:
                name = repr(blk)
            try:
                host = top.get_update_block_host_component(blk)
            except Exception:
                host = None
            try:
                src = inspect.getsource(blk)
            except (TypeError, OSError):
                src = None
            blk_info[blk] = {
                'name': name,
                'host': host,
                'src': src,
                'accesses': [],
            }

        # 2. Collect all CalleeIfcCL methods and their source.
        # NOTE: callee.method is a CalleePort wrapper, not the function.
        # The actual function is callee.method.method. Using callee.method
        # directly would make inspect.getsource fail (it would try to
        # fetch source for the CalleePort class, not the user method).
        #
        # After CLLineTracePass, callee.method.method is replaced with a
        # lambda wrapper. The original function is saved in
        # callee.method.raw_method (see CLLineTracePass.wrap_callee_method).
        # We prefer raw_method when available so the pass works correctly
        # even when run after CLLineTracePass.
        from pymtl3.dsl.Connectable import CalleeIfcCL, CallerIfcCL
        all_callees = top.get_all_object_filter(
            lambda x: isinstance(x, CalleeIfcCL)
        )
        callee_info = {}
        for callee in all_callees:
            try:
                port = callee.method
                if port is None:
                    continue
                # Prefer raw_method (original, pre-CLLineTracePass) over
                # method (which may be a lambda wrapper after CLLineTracePass).
                method = getattr(port, 'raw_method', None)
                if method is None:
                    method = port.method
                if method is None:
                    continue
                try:
                    src = inspect.getsource(method)
                except (TypeError, OSError):
                    src = None
                try:
                    host = callee.get_parent_object()
                except Exception:
                    host = None
                # Prefer the interface's declared name (e.g. "recv") over
                # the inner function's __name__ (e.g. "_recv_method") so
                # suggestion messages reference the user-facing attribute.
                name = (getattr(callee._dsl, 'my_name', None)
                        or getattr(method, '__name__', str(method)))
                callee_info[callee] = {
                    'name': name,
                    'host': host,
                    'src': src,
                    'accesses': [],
                    'method_obj': method,
                }
            except Exception as e:
                # Defensive: skip any callee whose structure we can't
                # introspect (e.g. dynamically generated wrappers).
                print(f"[PendingVarCheck] WARNING: could not collect callee "
                      f"{callee!r}: {e}", file=sys.stderr)
                continue

        # 3. Analyze each block/method with the AST visitor.
        # Each analysis is wrapped in try/except so that a single
        # block/method that the visitor cannot handle never crashes the
        # whole pass (this pass runs in DefaultPassGroup on EVERY model).
        for blk, info in blk_info.items():
            if info['src']:
                try:
                    visitor = _PendingVarVisitor()
                    accs = visitor.analyze(info['src'], info['name'])
                    info['accesses'] = accs
                except Exception as e:
                    print(f"[PendingVarCheck] WARNING: could not analyze "
                          f"block {info['name']}: {e}", file=sys.stderr)

        for callee, info in callee_info.items():
            if info['src']:
                try:
                    visitor = _PendingVarVisitor()
                    accs = visitor.analyze(info['src'], info['name'])
                    info['accesses'] = accs
                except Exception as e:
                    print(f"[PendingVarCheck] WARNING: could not analyze "
                          f"callee {info['name']}: {e}", file=sys.stderr)

        # 4. Build variable -> writers/readers map.
        # Use dicts keyed by the writer/reader object so that multiple
        # accesses within the same block/method collapse into a single
        # entry (e.g. two reads of s.pending_x in one block should only
        # produce one reader entry, not two).
        #
        # IMPORTANT: for M-kind (CalleeIfcCL) entries we key by and store
        # the underlying method FUNCTION (callee.method.method), not the
        # CalleeIfcCL wrapper. GenDAGPass stores M<U / M<M constraints
        # involving top-level callee ports in `top_level_callee_constraints`
        # as (method_func, blk_func) tuples -- the function identity, not
        # the wrapper, is what matches. Using the wrapper here would cause
        # false positives because `wrapper is func` is always False.
        var_writers = {}  # var_name -> dict {obj: (name, host, obj, kind)}
        var_readers = {}  # var_name -> dict {obj: (name, host, obj, kind)}

        for blk, info in blk_info.items():
            for acc in info['accesses']:
                if acc.access_type == "write":
                    var_writers.setdefault(acc.var_name, {})[blk] = (
                        info['name'], info['host'], blk, 'U'
                    )
                elif acc.access_type == "read":
                    var_readers.setdefault(acc.var_name, {})[blk] = (
                        info['name'], info['host'], blk, 'U'
                    )

        for callee, info in callee_info.items():
            method_obj = info['method_obj']  # the actual method function
            for acc in info['accesses']:
                if acc.access_type == "write":
                    var_writers.setdefault(acc.var_name, {})[method_obj] = (
                        info['name'], info['host'], method_obj, 'M'
                    )
                elif acc.access_type == "read":
                    var_readers.setdefault(acc.var_name, {})[method_obj] = (
                        info['name'], info['host'], method_obj, 'M'
                    )

        # 5. Check for missing constraints.
        # PyMTL3's GenDAGPass stores constraint tuples in two places:
        #   - `all_constraints`: U<U pairs (and M-pairs propagated through
        #     calling update blocks, i.e. when the callee is invoked from
        #     inside an @update block).
        #   - `top_level_callee_constraints`: (method_func, blk_or_method_func)
        #     pairs for M<U / M<M constraints where the callee is a top-level
        #     port (invoked by the simulator, not by an update block). These
        #     pairs do NOT appear in `all_constraints`, so we must consult
        #     both sets; otherwise every top-level CalleeIfcCL that shares a
        #     pending_* var with an @update block would be a false positive.
        all_constraints = top._dag.all_constraints
        top_level_callee_constraints = getattr(
            top._dag, 'top_level_callee_constraints', set()
        )

        # Build method_callers: method_func -> set of calling @update blocks.
        # This is needed to recognize propagated U<U constraints: when a child
        # component declares M(m) < U(r) and m is called from parent @update
        # block `caller`, GenDAGPass stores (caller, r) in all_constraints --
        # NOT (m, r). Without method_callers, we can't match (caller, r) to
        # the (m, r) pair we're checking, producing false positives.
        #
        # Strategy: match `call` objects from all_upblk_calls against the
        # CalleeIfcCL objects we already collected in callee_info (direct
        # identity match, no attribute access needed). For objects NOT in
        # callee_info (e.g. CallerIfcCL wired to a CalleeIfcCL, or raw
        # MethodPort), fall back to resolving via .method / .method.method
        # — which is safe after GenDAGPass has set CallerPort.method (line
        # 368 of GenDAGPass.py). All attribute access is wrapped in
        # try/except so a single unresolvable call never crashes the pass.
        method_callers = {}  # method_func -> set of blk

        # Fast lookup: CalleeIfcCL object -> its method function
        callee_obj_to_func = {}
        for callee, info in callee_info.items():
            callee_obj_to_func[callee] = info['method_obj']

        all_upblk_calls = getattr(top._dsl, 'all_upblk_calls', {})
        for blk, calls in all_upblk_calls.items():
            for call in calls:
                func = None
                # 1. Direct match: call is a CalleeIfcCL we collected
                if call in callee_obj_to_func:
                    func = callee_obj_to_func[call]
                else:
                    # 2. Fallback: resolve via attribute access (CallerIfcCL,
                    #    MethodPort, or other NonBlockingIfc/BlockingIfc).
                    #    After GenDAGPass, CallerPort.method is set to the
                    #    callee's function, so call.method.method gives the
                    #    actual method function.
                    #    NOTE: This is temporarily DISABLED because accessing
                    #    call.method.method on certain NonBlockingIfc objects
                    #    causes side effects that break ISA simulation. The
                    #    direct match above handles the common case (CalleeIfcCL
                    #    called directly in an update block). CallerIfcCL calls
                    #    are not resolved, which may cause some false positives
                    #    for propagated constraints — but those are warnings,
                    #    not errors, and the ISA test correctness is paramount.
                    pass
                if func is not None:
                    method_callers.setdefault(func, set()).add(blk)

        for var_name in var_writers:
            writers = list(var_writers[var_name].values())
            readers = list(var_readers.get(var_name, {}).values())

            for writer_name, writer_host, writer_obj, writer_kind in writers:
                for reader_name, reader_host, reader_obj, reader_kind in readers:
                    # Skip self-dependencies (same block reads and writes)
                    if writer_obj is reader_obj:
                        continue

                    # Skip cross-component false positives: variables with
                    # the same name in DIFFERENT sibling component instances
                    # are distinct variables (each component has its own
                    # s.pending_x). The visitor only tracks accesses via
                    # `s.<var>` where `s` is the component being analyzed, so
                    # all tracked accesses are intra-component. When two
                    # sibling components (e.g. IntIQCL and MemIQCL) both
                    # declare `s.pending_inserts`, they are different
                    # variables. Only writers and readers within the same
                    # host component could actually share a variable. If
                    # either host is None (introspection failed), fall
                    # through conservatively to avoid masking real issues.
                    if (writer_host is not None
                            and reader_host is not None
                            and writer_host is not reader_host):
                        continue

                    # Check if constraint already exists
                    has_constraint = self._has_constraint(
                        all_constraints, top_level_callee_constraints,
                        method_callers,
                        writer_obj, writer_kind,
                        reader_obj, reader_kind
                    )

                    if not has_constraint:
                        # Check for CL discipline violation
                        # (CalleeIfcCL writing state_*)
                        if writer_kind == 'M' and var_name.startswith(('state_', '_state_')):
                            violations.append(Violation(
                                kind="cl_discipline",
                                var_name=var_name,
                                writer_name=writer_name,
                                reader_name=reader_name,
                                message=(
                                    f"CL discipline violation: CalleeIfcCL "
                                    f"'{writer_name}' writes state_* variable "
                                    f"'{var_name}' directly (bypasses @update_ff). "
                                    f"Use pending_* instead."
                                ),
                            ))
                        elif writer_kind == 'M' and reader_kind == 'U':
                            violations.append(Violation(
                                kind="missing_m_u",
                                var_name=var_name,
                                writer_name=writer_name,
                                reader_name=reader_name,
                                message=(
                                    f"Missing M<U constraint: CalleeIfcCL "
                                    f"'{writer_name}' writes '{var_name}', "
                                    f"@update_once '{reader_name}' reads it. "
                                    f"Add: M(s.{writer_name}) < U({reader_name})"
                                ),
                                suggestion=(
                                    f"s.add_constraints(M(s.{writer_name}) < U({reader_name}))"
                                ),
                            ))
                        elif writer_kind == 'U' and reader_kind == 'U':
                            violations.append(Violation(
                                kind="missing_u_u",
                                var_name=var_name,
                                writer_name=writer_name,
                                reader_name=reader_name,
                                message=(
                                    f"Missing U<U constraint: @update_once "
                                    f"'{writer_name}' writes '{var_name}', "
                                    f"@update_once '{reader_name}' reads it. "
                                    f"Add: U({writer_name}) < U({reader_name})"
                                ),
                                suggestion=(
                                    f"s.add_constraints(U({writer_name}) < U({reader_name}))"
                                ),
                            ))
                        elif writer_kind == 'M' and reader_kind == 'M':
                            # M<M: check if callers have U<U
                            violations.append(Violation(
                                kind="missing_m_m",
                                var_name=var_name,
                                writer_name=writer_name,
                                reader_name=reader_name,
                                message=(
                                    f"Missing M<M or U<U constraint: CalleeIfcCL "
                                    f"'{writer_name}' writes '{var_name}', "
                                    f"CalleeIfcCL '{reader_name}' reads it. "
                                    f"Add M({writer_name}) < M({reader_name}), "
                                    f"or add U<U between their calling blocks."
                                ),
                            ))

        # 6. Print warnings or raise
        if violations:
            for v in violations:
                print(f"[PendingVarCheck] {v.kind}: {v.message}",
                      file=sys.stderr)
                if v.suggestion:
                    print(f"  Suggestion: {v.suggestion}", file=sys.stderr)

            if self.strict:
                raise Exception(
                    f"PendingVarCheckPass found {len(violations)} violation(s). "
                    f"Run without --strict to see warnings."
                )

        return violations

    def _has_constraint(self, all_constraints, top_level_callee_constraints,
                        method_callers,
                        writer_obj, writer_kind, reader_obj, reader_kind):
        """Check if a constraint exists between writer and reader in the
        constraint graph.

        PyMTL3's GenDAGPass stores M-involving constraints in two forms:

        1. `top_level_callee_constraints`: (method_func, blk_or_method_func)
           pairs for M<U / M<M constraints where the callee is a top-level
           port (invoked by the simulator, not by an update block).

        2. `all_constraints`: U<U pairs, INCLUDING M<U/M<M constraints that
           GenDAGPass has propagated through calling update blocks. When a
           child component declares M(m) < U(r) and m is called from
           @update block `caller`, GenDAGPass stores (caller, r) in
           all_constraints — NOT (m, r). To recognize these propagated
           constraints, we use `method_callers` (method_func -> set of
           calling blocks) to check whether any (caller, r) pair exists.

        Without the propagated-form check, every child-component
        CalleeIfcCL that shares a pending_* var with an @update_once in
        the SAME child component would be a false positive when the pass
        runs on the parent model (the constraint exists in the child's
        construct but is stored as propagated U<U, not as M<U).

        `writer_obj` / `reader_obj` for M-kind entries are the underlying
        method FUNCTIONS (callee.method.method), matching the identity used
        by GenDAGPass when it builds `top_level_callee_constraints` and
        `method_callers`.
        """
        # U<U: both are update blocks -- only stored in all_constraints.
        if writer_kind == 'U' and reader_kind == 'U':
            for (u, v) in all_constraints:
                if u is writer_obj and v is reader_obj:
                    return True
            return False

        # M<U, U<M, M<M: at least one side is a method.

        # 1. Direct M-involving constraint (top-level callee case).
        for (u, v) in top_level_callee_constraints:
            if u is writer_obj and v is reader_obj:
                return True

        # 2. Direct (method, blk) pair in all_constraints (rare, but
        #    check for completeness).
        for (u, v) in all_constraints:
            if u is writer_obj and v is reader_obj:
                return True

        # 3. Propagated U<U form: GenDAGPass converts M(m) < U(r) to
        #    U(caller) < U(r) where caller calls m. Check if any
        #    (caller, r) pair exists in all_constraints.
        writer_callers = method_callers.get(writer_obj)
        if writer_callers:
            for caller in writer_callers:
                for (u, v) in all_constraints:
                    if u is caller and v is reader_obj:
                        return True

        # 4. For M<M: GenDAGPass also converts M(m1) < M(m2) to
        #    U(caller1) < U(caller2). Check if any (caller1, caller2)
        #    pair exists in all_constraints.
        if writer_kind == 'M' and reader_kind == 'M':
            reader_callers = method_callers.get(reader_obj)
            if writer_callers and reader_callers:
                for caller_w in writer_callers:
                    for caller_r in reader_callers:
                        for (u, v) in all_constraints:
                            if u is caller_w and v is caller_r:
                                return True

        return False
