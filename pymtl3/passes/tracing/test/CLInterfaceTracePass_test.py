"""
#=========================================================================
# CLInterfaceTracePass_test.py
#=========================================================================
# Tests for CL interface text tracing extension in CLLineTracePass.
#
# These tests verify that CL interface method calls can be selectively
# logged as textual events with hierarchical paths, arguments, and return
# values, and that the feature is disabled by default.
"""

from io import StringIO

from pymtl3 import *

from ..CLLineTracePass import CLLineTracePass


#-------------------------------------------------------------------------
# Helper components
#-------------------------------------------------------------------------

class AccumulatorCL( Component ):
  """Simple component with a non-blocking add interface."""

  def construct( s ):
    s.total = 0

  @non_blocking( lambda s: True )
  def add( s, value ):
    s.total += int( value )
    return s.total


class GreeterCL( Component ):
  """Simple component with a blocking greet interface."""

  def construct( s ):
    pass

  @non_blocking( lambda s: True )
  def greet( s, name ):
    return f"hello {name}"


class TopCL( Component ):
  """Top component that exposes accumulator and greeter interfaces."""

  def construct( s ):
    s.acc    = AccumulatorCL()
    s.greeter = GreeterCL()

    s.add   = CallerIfcCL( Type=int   )
    s.greet = CallerIfcCL( Type=str   )

    s.add   //= s.acc.add
    s.greet //= s.greeter.greet


def _run_sim( top, n_ticks=3 ):
  top.elaborate()
  top.apply( DefaultPassGroup() )
  top.sim_reset()
  for _ in range( n_ticks ):
    if top.add.rdy():
      top.add( 5 )
    if top.greet.rdy():
      top.greet( "world" )
    top.sim_tick()


#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

def test_text_trace_disabled_by_default():
  """Without text_trace enabled, no events are emitted."""

  top = TopCL()
  sink = StringIO()
  top.set_metadata( CLLineTracePass.text_trace_output, sink.write )
  _run_sim( top )

  assert sink.getvalue() == ""


def test_text_trace_basic():
  """Enable text tracing and verify CALL events are emitted."""

  top = TopCL()
  sink = StringIO()
  top.set_metadata( CLLineTracePass.text_trace, True )
  top.set_metadata( CLLineTracePass.text_trace_output, sink )
  _run_sim( top )

  output = sink.getvalue()
  assert output != ""
  assert "CALL" in output
  assert "s.acc.add" in output
  assert "s.greeter.greet" in output
  assert "(5)" in output
  assert "(world)" in output


def test_text_trace_include_filter():
  """Only interfaces matching include patterns are traced."""

  top = TopCL()
  sink = StringIO()
  top.set_metadata( CLLineTracePass.text_trace, True )
  top.set_metadata( CLLineTracePass.text_trace_include, [ "*.acc.add" ] )
  top.set_metadata( CLLineTracePass.text_trace_output, sink )
  _run_sim( top )

  output = sink.getvalue()
  assert "s.acc.add" in output
  assert "s.greeter.greet" not in output


def test_text_trace_exclude_filter():
  """Excluded interfaces are omitted even if they match include."""

  top = TopCL()
  sink = StringIO()
  top.set_metadata( CLLineTracePass.text_trace, True )
  top.set_metadata( CLLineTracePass.text_trace_include, [ "*" ] )
  top.set_metadata( CLLineTracePass.text_trace_exclude, [ "*.acc.add" ] )
  top.set_metadata( CLLineTracePass.text_trace_output, sink )
  _run_sim( top )

  output = sink.getvalue()
  assert "s.acc.add" not in output
  assert "s.greeter.greet" in output


def test_text_trace_callable_output():
  """A plain callable can be used as output sink."""

  captured = []
  top = TopCL()
  top.set_metadata( CLLineTracePass.text_trace, True )
  top.set_metadata( CLLineTracePass.text_trace_include, [ "*.acc.add" ] )
  top.set_metadata( CLLineTracePass.text_trace_output, captured.append )
  _run_sim( top )

  assert any( "s.acc.add" in line for line in captured )


def test_text_trace_rdy():
  """Optionally trace rdy() calls separately."""

  top = TopCL()
  sink = StringIO()
  top.set_metadata( CLLineTracePass.text_trace, True )
  top.set_metadata( CLLineTracePass.text_trace_include, [ "*.acc.add" ] )
  top.set_metadata( CLLineTracePass.text_trace_rdy, True )
  top.set_metadata( CLLineTracePass.text_trace_output, sink )
  _run_sim( top )

  output = sink.getvalue()
  assert "CALL" in output
  assert "RDY" in output
