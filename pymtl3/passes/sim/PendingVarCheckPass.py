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
        # Variables written through aliases (e.g. `slot = s.state_x[i];
        # slot["field"] = val`). Kept SEPARATE from `accesses` so the
        # bidirectional check can detect alias writes without polluting
        # var_writers (which would create a false-positive explosion).
        self.alias_written_vars = set()

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
            # Track alias: b = s.state_x[i] (or b = s.state_x.field)
            # Records b -> state_x in _alias_map so subsequent b.field = val
            # or b[i] = val is recognized as a write to state_x (dict/list
            # element mutation through a local reference).
            #
            # Writes through aliases are recorded in `alias_written_vars`
            # (SEPARATE from `accesses`) so the bidirectional check can
            # detect them without polluting var_writers (which would create
            # a false-positive explosion from new writer→reader pairs).
            #
            # Aliases are invalidated when the variable is reassigned to a
            # non-tracked RHS, preventing stale aliases from producing
            # spurious writes.
            if isinstance(target, ast.Name):
                rhs = node.value
                if isinstance(rhs, (ast.Subscript, ast.Attribute)):
                    name_parts = self._get_full_name(rhs)
                    if name_parts:
                        tracked = self._is_tracked_name(name_parts)
                        if tracked:
                            self._alias_map[target.id] = tracked
                        else:
                            self._alias_map.pop(target.id, None)
                    else:
                        self._alias_map.pop(target.id, None)
                else:
                    self._alias_map.pop(target.id, None)
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
                    # to a tracked variable. Record in alias_written_vars
                    # (NOT accesses) to avoid polluting var_writers.
                    tracked_var = self._alias_map[name_parts[0]]
                    self.alias_written_vars.add(tracked_var)
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
                    # Indirect subscript write: b[i] = val where b is aliased
                    # to a tracked variable. Record in alias_written_vars.
                    tracked_var = self._alias_map[name_parts[0]]
                    self.alias_written_vars.add(tracked_var)

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
                    info['alias_written_vars'] = visitor.alias_written_vars
                except Exception as e:
                    print(f"[PendingVarCheck] WARNING: could not analyze "
                          f"block {info['name']}: {e}", file=sys.stderr)

        for callee, info in callee_info.items():
            if info['src']:
                try:
                    visitor = _PendingVarVisitor()
                    accs = visitor.analyze(info['src'], info['name'])
                    info['accesses'] = accs
                    info['alias_written_vars'] = visitor.alias_written_vars
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

        # Build obj -> set of alias-written var names.
        # Maps each block function (U kind) and method function (M kind) to
        # the set of tracked variables it writes through aliases (e.g.
        # `entry = s.state_lq[idx]; entry["field"] = val`). Used ONLY by the
        # bidirectional check to detect that the reader also writes the
        # variable, without polluting var_writers (which would create a
        # false-positive explosion from new writer→reader pairs).
        obj_to_alias_written_vars = {}
        for blk, info in blk_info.items():
            av = info.get('alias_written_vars')
            if av:
                obj_to_alias_written_vars[blk] = av
        for callee, info in callee_info.items():
            av = info.get('alias_written_vars')
            if av:
                obj_to_alias_written_vars[info['method_obj']] = av

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
        #
        #   - `_dsl.all_M_constraints`: the FULL set of user-declared M<M and
        #     M<U pairs, stored as (x, y, is_equal) triples where x/y can be
        #     CalleeIfcCL, CalleePort, CallerIfcCL, CallerPort, or raw
        #     function. GenDAGPass propagates these to `all_constraints`
        #     (in U<U form) ONLY when the callee is called from an
        #     @update_once block; otherwise (top-level callee or
        #     cross-component callee without a calling block) the M-pair
        #     stays ONLY in `all_M_constraints` and is invisible to the
        #     checks above. This produces false positives for every
        #     sub-component M<M/M<U constraint whose callee is not called
        #     from any @update_once block. To suppress these, we build a
        #     method_obj -> CalleeIfcCL/CalleePort reverse map and consult
        #     `all_M_constraints` directly.
        all_constraints = top._dag.all_constraints
        top_level_callee_constraints = getattr(
            top._dag, 'top_level_callee_constraints', set()
        )
        all_M_constraints = getattr(top._dsl, 'all_M_constraints', set())

        # Compute transitive closure of all_constraints (U<U edges only).
        # GenDAGPass does NOT close the constraint graph transitively, so a
        # registered-semantics pattern like U(A) < U(B) < U(C) does NOT
        # appear as (A, C) in all_constraints. We need the closure to detect
        # the "reader runs before writer" pattern (registered semantics) when
        # the reverse ordering is only transitively implied.
        #
        # Build adjacency list and BFS-reachable set per node. The graph is
        # small (typically < 500 nodes), so full closure is cheap.
        constraint_adj = {}  # node -> set of successor nodes
        for (u, v) in all_constraints:
            constraint_adj.setdefault(u, set()).add(v)
        constraint_reachable = {}  # node -> set of nodes reachable from it
        _all_nodes = set(constraint_adj.keys())
        for _s_set in constraint_adj.values():
            _all_nodes.update(_s_set)
        for _n in _all_nodes:
            _visited = set()
            _queue = [_n]
            while _queue:
                _cur = _queue.pop(0)
                for _nb in constraint_adj.get(_cur, ()):
                    if _nb not in _visited:
                        _visited.add(_nb)
                        _queue.append(_nb)
            constraint_reachable[_n] = _visited

        # Build method_obj -> {CalleeIfcCL, CalleePort} reverse maps so we
        # can match user-declared M<M / M<U constraints in
        # `all_M_constraints` against the (writer_obj, reader_obj) function
        # pairs the pass checks. The constraints store the original
        # CalleeIfcCL / CalleePort / CallerIfcCL / CallerPort objects, NOT
        # the resolved method function.
        from pymtl3.dsl.Connectable import (
            CalleePort, CallerPort, NonBlockingIfc, BlockingIfc
        )
        method_obj_to_callee_objs = {}  # method_func -> set of (callee_obj, port_obj)
        for callee, info in callee_info.items():
            try:
                port = callee.method
                mo = info['method_obj']
                method_obj_to_callee_objs.setdefault(mo, set()).add(
                    (callee, port)
                )
            except Exception:
                pass

        # Build method_callers: method_func -> set of calling @update blocks.
        # This is needed to recognize propagated U<U constraints: when a child
        # component declares M(m) < U(r) and m is called from parent @update
        # block `caller`, GenDAGPass stores (caller, r) in all_constraints --
        # NOT (m, r). Without method_callers, we can't match (caller, r) to
        # the (m, r) pair we're checking, producing false positives.
        #
        # The `call` objects in all_upblk_calls can be:
        #   - CalleeIfcCL (direct callee in an update block) -> match by
        #     identity against callee_info keys.
        #   - CallerIfcCL (caller interface in an update block) -> call.method
        #     is the CallerPort; we need to find which CalleeIfcCL it's
        #     connected to.
        #   - CallerPort (direct caller port in an update block) -> same
        #     resolution as CallerIfcCL.
        #
        # After CLLineTracePass, each CallerPort gets its OWN unique lambda
        # wrapper (wrap_caller_method creates a new lambda per CallerPort),
        # so function identity matching between CallerPort.method and
        # CalleePort.method/raw_method is impossible. Instead, we use
        # all_method_nets (from GenDAGPass) to build a CallerPort ->
        # CalleeIfcCL.method_obj mapping, leveraging the net topology.
        method_callers = {}  # method_func -> set of blk

        # Fast lookup: CalleeIfcCL object -> its method function
        callee_obj_to_func = {}
        for callee, info in callee_info.items():
            callee_obj_to_func[callee] = info['method_obj']

        # Build CallerPort -> raw_method mapping via all_method_nets.
        # Each method net is (writer_calleeport, {set of caller ports}).
        # The writer CalleePort is `callee.method` for some CalleeIfcCL;
        # we map each CallerPort in the net to that callee's raw_method.
        calleeport_to_func = {}
        for callee, info in callee_info.items():
            try:
                port = callee.method
                if isinstance(port, CalleePort):
                    calleeport_to_func[port] = info['method_obj']
            except Exception:
                pass

        callerport_to_func = {}  # CallerPort -> raw_method
        try:
            all_method_nets = top.get_all_method_nets()
            for writer, net in all_method_nets:
                raw = calleeport_to_func.get(writer)
                if raw is None:
                    continue
                for member in net:
                    if isinstance(member, CallerPort) and member is not writer:
                        callerport_to_func[member] = raw
        except Exception:
            pass

        all_upblk_calls = getattr(top._dsl, 'all_upblk_calls', {})
        for blk, calls in all_upblk_calls.items():
            for call in calls:
                func = None
                # 1. Direct match: call is a CalleeIfcCL we collected
                if call in callee_obj_to_func:
                    func = callee_obj_to_func[call]
                else:
                    # 2. Resolve CallerPort or CallerIfcCL to raw_method via
                    #    the callerport_to_func mapping built from method nets.
                    #    For CallerIfcCL: call.method is the CallerPort.
                    #    For CallerPort: call IS the CallerPort.
                    try:
                        if isinstance(call, CallerPort):
                            func = callerport_to_func.get(call)
                        else:
                            # CallerIfcCL or other NonBlockingIfc
                            cp = getattr(call, 'method', None)
                            if isinstance(cp, CallerPort):
                                func = callerport_to_func.get(cp)
                    except Exception:
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
                        all_M_constraints, method_obj_to_callee_objs,
                        constraint_reachable,
                        writer_obj, writer_kind,
                        reader_obj, reader_kind
                    )

                    # Bidirectional read-write detection (Trap #15 #3):
                    # When BOTH methods read AND write the same state variable,
                    # the pass may pick the wrong direction. If the forward
                    # constraint (writer → reader) is missing, check the REVERSE
                    # direction (reader → writer). If the reverse constraint
                    # exists, this is a false positive — the existing constraint
                    # covers the bidirectional case (registered semantics:
                    # reader runs before writer, reads OLD values; writer's
                    # new values visible next cycle).
                    if not has_constraint:
                        # Check if reader also writes and writer also reads
                        # the same variable (bidirectional access).
                        # Direct write: reader appears in var_writers for var.
                        # Alias write: reader writes via an alias like
                        # `entry = s.state_x[i]; entry["field"] = val`. These
                        # are tracked separately in obj_to_alias_written_vars
                        # to avoid polluting var_writers.
                        reader_also_writes = (
                            reader_obj in var_writers.get(var_name, {})
                            or var_name in obj_to_alias_written_vars.get(
                                reader_obj, set()
                            )
                        )
                        writer_also_reads = (
                            writer_obj in var_readers.get(var_name, {})
                        )
                        if reader_also_writes and writer_also_reads:
                            # Bidirectional: check reverse constraint.
                            # Applies to U<U and M<M (both methods read AND
                            # write the same var, so either direction could be
                            # correct depending on data flow).
                            if ((writer_kind == 'U' and reader_kind == 'U')
                                    or (writer_kind == 'M' and reader_kind == 'M')):
                                has_reverse = self._has_constraint(
                                    all_constraints,
                                    top_level_callee_constraints,
                                    method_callers,
                                    all_M_constraints, method_obj_to_callee_objs,
                                    constraint_reachable,
                                    reader_obj, reader_kind,
                                    writer_obj, writer_kind
                                )
                                if has_reverse:
                                    has_constraint = True
                                elif var_name.startswith(('state_', '_state_')):
                                    # Disjoint-access pattern (Trap #15 #4):
                                    # Both methods read AND write a state_*
                                    # variable via field mutation (dict entry
                                    # aliases like `entry = s.state_lq[idx];
                                    # entry["field"] = val`). No constraint
                                    # exists in EITHER direction.
                                    #
                                    # This is the NORMAL pattern for dict-based
                                    # state in CL models: multiple @update_once
                                    # blocks and CalleeIfcCL methods operate on
                                    # DIFFERENT fields of DIFFERENT entries,
                                    # guarded by fsm state or other conditions.
                                    # The designer intentionally left them
                                    # unordered because the operations are
                                    # disjoint or idempotent.
                                    #
                                    # Suppress to avoid false positives. If a
                                    # real data race exists (two methods writing
                                    # the SAME field of the SAME entry), it
                                    # should be caught by explicit testing, not
                                    # by this pass (which can't distinguish
                                    # disjoint from overlapping access).
                                    has_constraint = True
                        elif (writer_also_reads
                                and var_name.startswith(('state_', '_state_'))
                                and ((writer_kind == 'U' and reader_kind == 'U')
                                     or (writer_kind == 'M' and reader_kind == 'M'))):
                            # Dict-scan pattern (Trap #15 #5):
                            # Writer reads AND writes a state_* variable
                            # (guard-checked access via dict entry aliases like
                            # `entry = s.state_x[idx]; entry["field"] = val`),
                            # but reader ONLY reads (scans entries, e.g.
                            # `for idx in range(N): entry = s.state_x[idx];
                            # if entry["valid"]: ...`). No constraint exists
                            # in EITHER direction.
                            #
                            # This is the "writer initializes/updates specific
                            # entries, reader scans all entries" pattern,
                            # common in CL models with dict-based state. The
                            # writer's alias-assignment `entry = s.state_x[idx]`
                            # is a direct READ of state_x, and its subsequent
                            # field writes (e.g. `entry["valid"] = True`,
                            # `entry["has_stale_translation"] = False`) are
                            # INITIALIZATION writes that cause the reader to
                            # SKIP the entry (reader guards on `valid` and
                            # `has_stale_translation`).
                            #
                            # The race is benign: regardless of writer/reader
                            # ordering, the reader skips the entry being
                            # initialized (if reader runs first, entry is
                            # `valid=False` -> skip; if reader runs after,
                            # entry is `valid=True, has_stale_translation=False`
                            # -> skip). No constraint is needed.
                            #
                            # Suppress to avoid false positives. Real races on
                            # overlapping DATA fields of the SAME entry (where
                            # the reader's behavior depends on the writer's
                            # value) should be caught by explicit testing.
                            has_constraint = True

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
                        all_M_constraints, method_obj_to_callee_objs,
                        constraint_reachable,
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

        3. `all_M_constraints`: the FULL set of user-declared M<M and M<U
           pairs (not propagated), stored as (x, y, is_equal) triples
           where x/y can be CalleeIfcCL, CalleePort, CallerIfcCL,
           CallerPort, or raw function. GenDAGPass propagates these to
           `all_constraints` (in U<U form) ONLY when the callee is called
           from an @update_once block; otherwise the M-pair stays ONLY in
           `all_M_constraints`. We consult this set directly via a
           method_obj -> {callee_obj, port_obj} reverse map to recognize
           user-declared constraints that the propagated-form check misses.
        """
        # U<U: both are update blocks -- only stored in all_constraints.
        if writer_kind == 'U' and reader_kind == 'U':
            # Forward constraint: writer < reader
            for (u, v) in all_constraints:
                if u is writer_obj and v is reader_obj:
                    return True
            # Reverse constraint (registered semantics): reader runs before
            # writer transitively. GenDAGPass does NOT transitively close
            # all_constraints, so we use the pre-computed BFS closure.
            #
            # This handles the parent-child ordering pattern where the reader
            # reads OLD values (from previous cycle) and the writer's new
            # values are visible next cycle. Without this check, every
            # U<U pair without a direct forward constraint would be flagged,
            # even when the reverse ordering is intentionally established via
            # transitive constraints (e.g. U(reader) < U(parent.up_process)
            # < U(writer) via M<U propagation).
            if (reader_obj in constraint_reachable
                    and writer_obj in constraint_reachable.get(reader_obj, ())):
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

        # 5. Direct lookup in `all_M_constraints`. This catches all
        #    user-declared M<M / M<U constraints whose CalleeIfcCL is NOT
        #    called from any @update_once block (so GenDAGPass never
        #    propagated them to `all_constraints`). This is the primary
        #    mechanism for recognizing top-level callee constraints and
        #    sub-component constraints that the propagated-form checks
        #    above cannot see.
        #
        # `all_M_constraints` stores (x, y, is_equal) triples where x/y
        # can be CalleeIfcCL, CalleePort, CallerIfcCL, CallerPort, or raw
        # function. For M<U, y is the @update_once block function of the
        # SAME component instance that owns x. When the model has multiple
        # component instances (e.g. per-thread), each has its own
        # `up_dispatch` function object, so we must match y against
        # reader_obj by identity OR by (host, name) when both are update
        # blocks of the same component.
        writer_callee_objs = method_obj_to_callee_objs.get(writer_obj, set())
        reader_callee_objs = method_obj_to_callee_objs.get(reader_obj, set())
        # Also include the method_obj itself in case the constraint
        # was declared with the raw function rather than a CalleeIfcCL.
        writer_candidates = set(writer_callee_objs)
        writer_candidates.add((writer_obj, writer_obj))
        if reader_kind == 'M':
            reader_candidates = set(reader_callee_objs)
            reader_candidates.add((reader_obj, reader_obj))
        else:
            # M<U case: reader is an @update_once block; reader_obj is
            # the block function. M<U constraints store the U side as a
            # function directly.
            reader_candidates = {(reader_obj, reader_obj)}

        # For M<U, also resolve reader_obj's host and name so we can
        # match y by (host, name) when identity fails (multiple instances
        # of the same block name in different components).
        reader_host = None
        reader_name = None
        if reader_kind == 'U':
            try:
                reader_host = top.get_update_block_host_component(reader_obj)
            except Exception:
                reader_host = None
            reader_name = getattr(reader_obj, '__name__', None)

        for (w_callee, w_port) in writer_candidates:
            # Resolve writer's host for (host, name) matching.
            w_host = None
            try:
                # w_callee may be a CalleeIfcCL or a function.
                if hasattr(w_callee, 'get_parent_object'):
                    w_host = w_callee.get_parent_object()
            except Exception:
                w_host = None
            for (r_callee, r_port) in reader_candidates:
                for (x, y, is_equal) in all_M_constraints:
                    if is_equal:
                        continue
                    # Match x against any form of the writer (CalleeIfcCL,
                    # CalleePort, or raw function).
                    x_match = (x is w_callee or x is w_port
                               or x is writer_obj)
                    if not x_match:
                        continue
                    # Match y against any form of the reader.
                    y_match = (y is r_callee or y is r_port
                               or y is reader_obj)
                    if y_match:
                        return True
                    # M<U fallback: match by (host, name) when both y and
                    # reader_obj are update blocks of the same component.
                    if (reader_kind == 'U' and reader_host is not None
                            and reader_name is not None
                            and w_host is not None):
                        y_name = getattr(y, '__name__', None)
                        if y_name is not None and y_name == reader_name:
                            try:
                                y_host = top.get_update_block_host_component(y)
                            except Exception:
                                y_host = None
                            if y_host is w_host:
                                return True

        # 6. Registered-semantics (reverse) check.
        # If the REVERSE ordering exists (reader runs BEFORE writer, possibly
        # transitively), the user intentionally chose registered semantics:
        # the reader reads OLD values and the writer's new values are visible
        # next cycle. The forward constraint (writer < reader) would create a
        # cycle and is intentionally omitted.
        #
        # GenDAGPass does NOT transitively close all_constraints, so we use
        # `constraint_reachable` (pre-computed BFS closure) to detect
        # transitive reverse orderings like:
        #   U(reader) < U(parent.up_process) < U(writer)
        # which arise from parent-child ordering + M<U propagation.
        #
        # U<U case: reader reaches writer transitively.
        if writer_kind == 'U' and reader_kind == 'U':
            if (reader_obj in constraint_reachable
                    and writer_obj in constraint_reachable.get(reader_obj, ())):
                return True
            # Debug: check why reverse check fails
            import os
            if os.environ.get('PVCP_DEBUG2'):
                r_name = getattr(reader_obj, '__name__', '?')
                w_name = getattr(writer_obj, '__name__', '?')
                r_in = reader_obj in constraint_reachable
                w_in_reach = writer_obj in constraint_reachable.get(reader_obj, set())
                # Check if reader_obj is in constraint_adj at all
                r_in_adj = reader_obj in constraint_adj if 'constraint_adj' in dir() else '?'
                print(f"  RVCHK U<U: reader={r_name}(id={id(reader_obj)}) writer={w_name}(id={id(writer_obj)}) "
                      f"r_in_reachable={r_in} w_in_r_reach={w_in_reach}", file=__import__('sys').stderr)
                # Show what reader_obj CAN reach
                if r_in:
                    reach = constraint_reachable[reader_obj]
                    for n in list(reach)[:5]:
                        n_name = getattr(n, '__name__', repr(n))
                        print(f"    reach: {n_name}(id={id(n)})", file=__import__('sys').stderr)

        # M<U case: reader reaches a caller of the writer transitively.
        # This is the parent-child ordering pattern: the parent's up_process
        # calls the child's CalleeIfcCL (writer), and the child's up_process
        # (reader) is constrained to run before the parent's up_process.
        #   U(reader) < U(parent.up_process)  [parent-child ordering]
        #   parent.up_process calls writer    [M<U propagation]
        # So reader runs before writer — registered semantics.
        if writer_kind == 'M' and reader_kind == 'U':
            writer_callers = method_callers.get(writer_obj)
            if writer_callers:
                reader_reach = constraint_reachable.get(reader_obj, ())
                for caller in writer_callers:
                    if caller in reader_reach:
                        return True

        # M<M case: a caller of the reader reaches a caller of the writer
        # transitively (reverse M<M via propagated U<U).
        if writer_kind == 'M' and reader_kind == 'M':
            writer_callers = method_callers.get(writer_obj, set())
            reader_callers = method_callers.get(reader_obj, set())
            for r_caller in reader_callers:
                r_reach = constraint_reachable.get(r_caller, ())
                for w_caller in writer_callers:
                    if w_caller in r_reach:
                        return True

        return False
