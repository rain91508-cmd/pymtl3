#=========================================================================
# DynamicSchedulePass.py
#=========================================================================
#
# Author : Shunning Jiang
# Date   : Apr 19, 2019

import os
from collections import defaultdict, deque
from copy import deepcopy

import py

from pymtl3.datatypes import Bits, is_bitstruct_class
from pymtl3.dsl import CalleePort
from pymtl3.dsl.errors import UpblkCyclicError
from pymtl3.extra.pypy import custom_exec
from pymtl3.passes.BasePass import BasePass, PassMetadata
from pymtl3.passes.errors import PassOrderError

from .SimpleSchedulePass import SimpleSchedulePass, dump_dag
from .SimpleTickPass import SimpleTickPass


# ---------------------------------------------------------------------------
# Deterministic scheduling helpers (shared with SimpleSchedulePass).
# These functions produce stable sort keys that are independent of id(),
# memory address, or set iteration order, making the schedule reproducible
# across runs regardless of PYTHONHASHSEED or process ID.
# ---------------------------------------------------------------------------

def _stable_src_loc( v ):
  """Stable source-location identifier (independent of id()/memory)."""
  func = None
  if isinstance( v, CalleePort ):
    func = getattr( v, "method", None )
  else:
    func = v
  if func is not None and hasattr( func, "__code__" ):
    co = func.__code__
    return ( co.co_filename, co.co_firstlineno )
  return ( "", 0 )

def _stable_func_name( v ):
  """Stable function name (never includes id/memory address)."""
  if isinstance( v, CalleePort ):
    return repr( v )
  nm = getattr( v, "__name__", None )
  if nm is not None:
    return nm
  return f"<unknown:{type(v).__name__}>"

def _stable_host_repr( v, top ):
  """Stable host component repr (empty string for internal blocks)."""
  try:
    if isinstance( v, CalleePort ):
      host = v.get_parent_object()
    else:
      host = top.get_update_block_host_component( v )
    return repr( host )
  except Exception:
    return ""

def _make_stable_sort_key( top ):
  """Return a sort key function bound to `top` for host lookup."""
  def _key( v ):
    return ( _stable_host_repr( v, top ), _stable_func_name( v ), _stable_src_loc( v ) )
  return _key


class DynamicSchedulePass( BasePass ):
  def __call__( self, top ):
    if not hasattr( top._dag, "all_constraints" ):
      raise PassOrderError( "all_constraints" )

    if hasattr( top, "_sched" ):
      raise Exception("Some schedule pass has already been applied!")

    top._sched = PassMetadata()

    self.schedule_intra_cycle( top )

    # Reuse simple's ff and flip schedule
    simple = SimpleSchedulePass()
    simple.schedule_ff( top )
    simple.schedule_posedge_flip( top )

  def schedule_intra_cycle( self, top ):

    # Construct the intra-cycle graph based on normal update blocks

    V   = top._dag.final_upblks - top.get_all_update_ff()

    import os
    _deterministic = os.environ.get( "PYMTL3_DETERMINISTIC_SCHED", "" )
    _perm_seed = os.environ.get( "PYMTL3_SCHED_PERM_SEED", "" )

    # A permutation seed only makes sense on top of a reproducible base order,
    # so setting it implicitly forces the deterministic branch. It then drives
    # seeded tie-breaking in the topological sort so every seed yields a
    # *different but reproducible* valid schedule order. This lets a
    # schedule-dependent bug be reproduced and bisected across seeds.
    if _deterministic or _perm_seed:
      # Deterministic scheduling: use a stable sort key to make the schedule
      # reproducible across runs. Set iteration order depends on id()/hash of
      # function objects, which varies across processes. Sorting by a stable
      # key (host_repr, func_name, src_loc) eliminates this source of
      # nondeterminism.  Opt-in via PYMTL3_DETERMINISTIC_SCHED=1.
      _sort_key = _make_stable_sort_key( top )
      V_sorted = sorted( V, key=_sort_key )

      if _perm_seed != "":
        import random as _random
        _rng = _random.Random( int( _perm_seed ) )
        _seeded = True
        _tmp = list( V_sorted )
        _rng.shuffle( _tmp )
        V_sorted = _tmp
      else:
        _rng = None
        _seeded = False

      G   = { v: [] for v in V_sorted }
      G_T = { v: [] for v in V_sorted } # transpose graph

      E = set()
      for (u, v) in top._dag.all_constraints: # u -> v
        if u in V and v in V:
          G  [u].append( v )
          G_T[v].append( u )
          E.add( (u, v) )

      if 'MAMBA_DAG' in os.environ:
        dump_dag( top, V, E )

      SCCs, G_new = kosaraju_scc( G, G_T, _sort_key )

      scc_sort_keys = []
      for scc in SCCs:
        min_key = min( _sort_key( v ) for v in scc )
        scc_sort_keys.append( min_key )

      InD = { i: 0 for i in range(len(SCCs)) }
      for u, vs in G_new.items():
        for v in vs:
          InD[ v ] += 1

      scc_pred = {}
      scc_schedule = []

      # Seeded tie-break: pick a random ready SCC instead of the min-key one,
      # so the top-level order varies per seed while staying topologically
      # valid.  Non-seeded keeps the original min-key (stable) behaviour.
      ready = [ i for i in range(len(SCCs)) if not InD[i] ]
      for i in ready:
        scc_pred[ i ] = None

      while ready:
        if _seeded:
          j = _rng.randrange( len( ready ) )
          u = ready.pop( j )
        else:
          u = min( ready, key=lambda i: scc_sort_keys[i] )
          ready.remove( u )
        scc_schedule.append( u )
        for v in sorted( G_new[u], key=lambda x: scc_sort_keys[x] ):
          InD[v] -= 1
          if not InD[v]:
            ready.append( v )
            scc_pred[ v ] = u

      assert len(scc_schedule) == len(SCCs)

      constraint_objs = top._dag.constraint_objs
      onces = top.get_all_update_once()

      top._sched.update_schedule = schedule = []

      scc_id = 0
      for i in scc_schedule:
        scc = SCCs[i]
        if len(scc) == 1:
          schedule.append( list(scc)[0] )
        else:
          for x in scc:
            if x in onces:
              raise UpblkCyclicError("update_once blocks are not allowed to appear in a cycle. \n - " + \
                              "\n - ".join( [
                                f"{y.__name__} ({'@update_once' if y in onces else '@update'} " \
                                f"in 'top.{repr(top.get_update_block_host_component(y))[2:]}')"
                                for y in scc] ))

          tmp_schedule = []
          Q = deque()

          if scc_pred[i] is None:
            InD = { v: 0 for v in scc }
            for (u, v) in E:
              if u in scc and v in scc:
                InD[ v ] += 1
            _max_in = max( InD.values() )
            _cands = [ v for v in scc if InD[v] == _max_in ]
            if _seeded and len(_cands) > 1:
              root = _cands[ _rng.randrange( len(_cands) ) ]
            else:
              root = max( _cands, key=lambda v: (InD[v], _sort_key(v)) )
            Q.append( root )
          else:
            pred = set( SCCs[ scc_pred[i] ] )
            for x in sorted( scc, key=_sort_key ):
              for v in G_T[x]:
                if v in pred:
                  Q.append( x )

          visited = set(Q)
          while Q:
            if _seeded and len(Q) > 1:
              j = _rng.randrange( len(Q) )
              u = Q[j]; del Q[j]
            else:
              u = Q.popleft()
            tmp_schedule.append( u )
            for v in sorted( G[u], key=_sort_key ):
              if v in scc and v not in visited:
                Q.append( v )
                visited.add( v )

          scc_id += 1
          variables = set()
          for (u, v) in E:
            if u in scc and v in scc:
              variables.update( constraint_objs[ (u, v) ] )

          if len(variables) == 0:
            raise UpblkCyclicError("There is a cyclic dependency without involving variables."
                            "Probably a loop that involves blocks that should be update_once:\n{}"\
                            .format(", ".join( [ x.__name__ for x in scc] )))

          def gen_wrapped_SCCblk( s, scc, src ):
            scc_tick_func = SimpleTickPass.gen_tick_function( scc )
            _globals = { 's': s, 'scc_tick_func': scc_tick_func, 'deepcopy': deepcopy,
                         'UpblkCyclicError': UpblkCyclicError }
            _locals  = {}
            custom_exec(py.code.Source( src ).compile(), _globals, _locals)
            return _locals[ 'generated_block' ]

          template = """
def wrapped_SCC_{0}():
  N = 0
  while True:
    N += 1
    if N > 100:
      raise UpblkCyclicError("Combinational loop detected at runtime in {{{3}}} after 100 iters!")
    {1}
    scc_tick_func()
    {2}
    break
generated_block = wrapped_SCC_{0}
          """

          copy_srcs  = []
          check_srcs = []

          final_variables = set()
          for x in sorted( variables, key=repr ):
            w = x.get_top_level_signal()
            if w is x:
              final_variables.add( x )
              continue
            if issubclass( w._dsl.Type, Bits ):
              if w not in final_variables:
                final_variables.add( w )
            elif is_bitstruct_class( w._dsl.Type ):
              if w not in final_variables:
                final_variables.add( x )
            else:
              final_variables.add( x )

          final_var_host = defaultdict(list)
          for x in final_variables:
            final_var_host[ x.get_host_component() ].append( x )

          var_id = 0
          for host, var_list in final_var_host.items():
            copy_srcs .append( f"host={host!r}" )
            check_srcs.append( f"host={host!r}" )
            sub_check_srcs = []
            hostlen = len(repr(host))
            for var in var_list:
              var_id += 1
              subname = repr(var)[hostlen+1:]
              if issubclass( var._dsl.Type, Bits ):
                copy_srcs.append( f"t{var_id}=host.{subname}.clone()" )
              elif is_bitstruct_class( var._dsl.Type ):
                copy_srcs.append( f"t{var_id}=host.{subname}.clone()" )
              else:
                copy_srcs.append( f"t{var_id}=deepcopy(host.{subname})" )
              sub_check_srcs.append( f"host.{subname} != t{var_id}" )
            check_srcs.append( f"if { ' or '.join(sub_check_srcs)}: continue" )

          scc_block_src = template.format( scc_id, "; ".join( copy_srcs ), "\n    ".join( check_srcs ),
                                           ", ".join( [ x.__name__ for x in scc] ) )
          _gen = gen_wrapped_SCCblk( top, tmp_schedule, scc_block_src )
          # Label the generated SCC block with its member order so schedule
          # dumps are human-diffable across seeds.
          _gen.__name__ = "SCC%d__" % scc_id + ",".join(
              getattr(x, "__name__", repr(x)) for x in tmp_schedule )
          schedule.append( _gen )

    else:
      # Non-deterministic (default): rely on set/dict iteration order.
      G   = { v: [] for v in V }
      G_T = { v: [] for v in V }

      E = set()
      for (u, v) in top._dag.all_constraints:
        if u in V and v in V:
          G  [u].append( v )
          G_T[v].append( u )
          E.add( (u, v) )

      if 'MAMBA_DAG' in os.environ:
        dump_dag( top, V, E )

      SCCs, G_new = kosaraju_scc( G, G_T )

      InD = { i: 0 for i in range(len(SCCs)) }
      for u, vs in G_new.items():
        for v in vs:
          InD[ v ] += 1

      scc_pred = {}
      scc_schedule = []

      Q = deque( [ i for i in range(len(SCCs)) if not InD[i] ] )
      for x in Q:
        scc_pred[ x ] = None

      while Q:
        u = Q.pop()
        scc_schedule.append( u )
        for v in G_new[u]:
          InD[v] -= 1
          if not InD[v]:
            Q.append( v )
            scc_pred[ v ] = u

      assert len(scc_schedule) == len(SCCs)

      constraint_objs = top._dag.constraint_objs
      onces = top.get_all_update_once()

      top._sched.update_schedule = schedule = []

      scc_id = 0
      for i in scc_schedule:
        scc = SCCs[i]
        if len(scc) == 1:
          schedule.append( list(scc)[0] )
        else:
          for x in scc:
            if x in onces:
              raise UpblkCyclicError("update_once blocks are not allowed to appear in a cycle. \n - " + \
                              "\n - ".join( [
                                f"{y.__name__} ({'@update_once' if y in onces else '@update'} " \
                                f"in 'top.{repr(top.get_update_block_host_component(y))[2:]}')"
                                for y in scc] ))

          tmp_schedule = []
          Q = deque()

          if scc_pred[i] is None:
            InD = { v: 0 for v in scc }
            for (u, v) in E:
              if u in scc and v in scc:
                InD[ v ] += 1
            Q.append( max(InD, key=InD.get) )
          else:
            pred = set( SCCs[ scc_pred[i] ] )
            for x in scc:
              for v in G_T[x]:
                if v in pred:
                  Q.append( x )

          visited = set(Q)
          while Q:
            u = Q.popleft()
            tmp_schedule.append( u )
            for v in G[u]:
              if v in scc and v not in visited:
                Q.append( v )
                visited.add( v )

          scc_id += 1
          variables = set()
          for (u, v) in E:
            if u in scc and v in scc:
              variables.update( constraint_objs[ (u, v) ] )

          if len(variables) == 0:
            raise UpblkCyclicError("There is a cyclic dependency without involving variables."
                            "Probably a loop that involves blocks that should be update_once:\n{}"\
                            .format(", ".join( [ x.__name__ for x in scc] )))

          def gen_wrapped_SCCblk( s, scc, src ):
            scc_tick_func = SimpleTickPass.gen_tick_function( scc )
            _globals = { 's': s, 'scc_tick_func': scc_tick_func, 'deepcopy': deepcopy,
                         'UpblkCyclicError': UpblkCyclicError }
            _locals  = {}
            custom_exec(py.code.Source( src ).compile(), _globals, _locals)
            return _locals[ 'generated_block' ]

          template = """
def wrapped_SCC_{0}():
  N = 0
  while True:
    N += 1
    if N > 100:
      raise UpblkCyclicError("Combinational loop detected at runtime in {{{3}}} after 100 iters!")
    {1}
    scc_tick_func()
    {2}
    break
generated_block = wrapped_SCC_{0}
          """

          copy_srcs  = []
          check_srcs = []

          final_variables = set()
          for x in sorted( variables, key=repr ):
            w = x.get_top_level_signal()
            if w is x:
              final_variables.add( x )
              continue
            if issubclass( w._dsl.Type, Bits ):
              if w not in final_variables:
                final_variables.add( w )
            elif is_bitstruct_class( w._dsl.Type ):
              if w not in final_variables:
                final_variables.add( x )
            else:
              final_variables.add( x )

          final_var_host = defaultdict(list)
          for x in final_variables:
            final_var_host[ x.get_host_component() ].append( x )

          var_id = 0
          for host, var_list in final_var_host.items():
            copy_srcs .append( f"host={host!r}" )
            check_srcs.append( f"host={host!r}" )
            sub_check_srcs = []
            hostlen = len(repr(host))
            for var in var_list:
              var_id += 1
              subname = repr(var)[hostlen+1:]
              if issubclass( var._dsl.Type, Bits ):
                copy_srcs.append( f"t{var_id}=host.{subname}.clone()" )
              elif is_bitstruct_class( var._dsl.Type ):
                copy_srcs.append( f"t{var_id}=host.{subname}.clone()" )
              else:
                copy_srcs.append( f"t{var_id}=deepcopy(host.{subname})" )
              sub_check_srcs.append( f"host.{subname} != t{var_id}" )
            check_srcs.append( f"if { ' or '.join(sub_check_srcs)}: continue" )

          scc_block_src = template.format( scc_id, "; ".join( copy_srcs ), "\n    ".join( check_srcs ),
                                           ", ".join( [ x.__name__ for x in scc] ) )
          _gen = gen_wrapped_SCCblk( top, tmp_schedule, scc_block_src )
          # Label the generated SCC block with its member order so schedule
          # dumps are human-diffable across seeds.
          _gen.__name__ = "SCC%d__" % scc_id + ",".join(
              getattr(x, "__name__", repr(x)) for x in tmp_schedule )
          schedule.append( _gen )

def kosaraju_scc( G, G_T, sort_key=None ):

    #---------------------------------------------------------------------
    # Run Kosaraju's algorithm to shrink all strongly connected components
    # (SCCs) into super nodes
    #---------------------------------------------------------------------

    # First dfs on G to generate reverse post-order (RPO)
    # Shunning: we emulate the system stack to implement non-recursive
    # post-order DFS algorithm. At the beginning, I implemented a more
    # succinct recursive DFS but it turned out that a 1500-depth chain in
    # the graph will reach the CPython max recursion depth.
    # https://docs.python.org/3/library/sys.html#sys.getrecursionlimit

    PO = []

    # Deterministic: sort vertices by stable key instead of relying on
    # dict/set iteration order (which depends on id()/hash of functions).
    if sort_key is not None:
      vertices = sorted( G.keys(), key=sort_key )
    else:
      vertices = list(G.keys())
    visited = set()

    # The following algorithm push all adjacent elements to the stack at
    # once and later check visited set to avoid redundant visit (instead
    # of checking visited set when pushing element to the stack). I added
    # a second_visit flag to add the node to post-order.

    for u in vertices:
      stack = [ (u, False) ]
      while stack:
        u, second_visit = stack.pop()

        if second_visit:
          PO.append( u )
        elif u not in visited:
          visited.add( u )
          stack.append( (u, True) )
          # Deterministic: sort adjacent vertices by stable key before
          # pushing to stack. reversed() + sorted = descending sort key.
          adj = G[u]
          if sort_key is not None:
            adj = sorted( adj, key=sort_key, reverse=True )
          else:
            adj = reversed( adj )
          for v in adj:
            stack.append( (v, False) )

    RPO = PO[::-1]

    # Second bfs on G_T to generate SCCs

    SCCs  = []
    v_SCC = {}
    visited = set()

    for u in RPO:
      if u not in visited:
        visited.add( u )
        scc = set()
        SCCs.append( scc )
        Q = deque( [u] )
        scc.add( u )
        while Q:
          u = Q.popleft()
          v_SCC[u] = len(SCCs) - 1
          # Deterministic: sort G_T[u] by stable key
          adj = G_T[u]
          if sort_key is not None:
            adj = sorted( adj, key=sort_key )
          for v in adj:
            if v not in visited:
              visited.add( v )
              Q.append( v )
              scc.add( v )

    # Construct a new graph of SCCs

    G_new = { i: set() for i in range(len(SCCs)) }

    for u, vs in G.items():
      for v in vs: # u -> v
        scc_u, scc_v = v_SCC[u], v_SCC[v]
        if scc_u != scc_v and scc_v not in G_new[ scc_u ]:
          G_new[ scc_u ].add( scc_v )

    return SCCs, G_new
