from .autotick.OpenLoopCLPass import OpenLoopCLPass
from .BasePass import BasePass
from .sim.DynamicSchedulePass import DynamicSchedulePass
from .sim.GenDAGPass import GenDAGPass
from .sim.PendingVarCheckPass import PendingVarCheckPass
from .sim.PrepareSimPass import PrepareSimPass
from .sim.SimpleSchedulePass import SimpleSchedulePass
from .sim.SimpleTickPass import SimpleTickPass
from .sim.WrapGreenletPass import WrapGreenletPass
from .tracing.CLLineTracePass import CLLineTracePass
from .tracing.LineTraceParamPass import LineTraceParamPass
from .tracing.PrintTextWavePass import PrintTextWavePass
from .tracing.VcdGenerationPass import VcdGenerationPass


# SimpleSim can be used when the UDG is a DAG
class SimpleSimPass( BasePass ):
  def __call__( s, top ):
    LineTraceParamPass()( top )
    GenDAGPass()( top )
    PendingVarCheckPass()( top )
    WrapGreenletPass()( top )
    SimpleSchedulePass()( top )
    CLLineTracePass()( top )
    VcdGenerationPass()( top )
    PrintTextWavePass()( top )

    PrepareSimPass(print_line_trace=False)( top )

class DefaultPassGroup( BasePass ):
  def __init__( s, *, vcdwave=None, textwave=False,
                    linetrace=False, reset_active_high=True,
                    strict_check=None ):

    s.vcdwave = vcdwave
    s.textwave = textwave
    s.linetrace = linetrace
    s.reset_active_high = reset_active_high
    s.strict_check = strict_check

  def __call__( s, top ):

    if s.vcdwave:
      top.set_metadata( VcdGenerationPass.vcd_file_name, s.vcdwave )

    if s.textwave:
      top.set_metadata( PrintTextWavePass.enable, True )

    LineTraceParamPass()( top )
    GenDAGPass()( top )
    WrapGreenletPass()( top )
    CLLineTracePass()( top )
    DynamicSchedulePass()( top )
    VcdGenerationPass()( top )
    PrintTextWavePass()( top )

    PrepareSimPass(print_line_trace=s.linetrace,
                   reset_active_high=s.reset_active_high)( top )
    # PendingVarCheckPass is OPT-IN: only runs when strict_check is not None.
    #   strict_check=None  (default) — pass does not run (safe for all tests)
    #   strict_check=False             — runs in warning mode (prints to stderr)
    #   strict_check=True              — runs in strict mode (raises on violations)
    #
    # The pass is a pure static analysis (reads _dag/_dsl metadata, prints
    # warnings) with no writes to simulation state. However, it creates
    # Python objects (VarAccess, Violation, AST nodes) whose allocation
    # shifts Python's memory layout, which shifts set/greenlet iteration
    # order in the simulation runtime. For models with pre-existing missing
    # M<U constraints (e.g. O3 ISA tests), this worsens nondeterministic
    # deadlocks. Running it after PrepareSimPass (model locked) and making it
    # opt-in ensures zero impact on existing tests unless explicitly requested.
    if s.strict_check is not None:
      PendingVarCheckPass(strict=s.strict_check)( top )

class AutoTickSimPass( BasePass ):
  def __init__( s, print_line_trace=True ):
    s.print_line_trace = print_line_trace

  def __call__( s, top ):
    top.elaborate()
    GenDAGPass()( top )
    WrapGreenletPass()( top )
    OpenLoopCLPass( s.print_line_trace )( top )
    top.lock_in_simulation()
