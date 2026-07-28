#========================================================================
# CLLineTracePass.py
#========================================================================
# Enable CL line trace.
#
# Author : Yanghui Ou
#   Date : May 21, 2019

import fnmatch
import re
import sys

from pymtl3.dsl import *
from pymtl3.passes.BasePass import BasePass


class CLLineTracePass( BasePass ):

  # CLLineTracePass public pass data

  #: enable
  #:
  #: Type: ``bool``; input
  #:
  #: Default value: True
  enable = MetadataKey(bool)

  clear_cl_trace_func = MetadataKey()

  #: text_trace
  #:
  #: Type: ``bool``; input
  #:
  #: Default value: False
  text_trace = MetadataKey(bool)

  #: text_trace_include
  #:
  #: Type: ``list[str]``; input
  #:
  #: Default value: ["*"]
  text_trace_include = MetadataKey()

  #: text_trace_exclude
  #:
  #: Type: ``list[str]``; input
  #:
  #: Default value: []
  text_trace_exclude = MetadataKey()

  #: text_trace_output
  #:
  #: Type: ``callable`` or ``TextIO``; input
  #:
  #: Default value: sys.stderr.write
  text_trace_output = MetadataKey()

  #: text_trace_max_len
  #:
  #: Type: ``int``; input
  #:
  #: Default value: 120
  text_trace_max_len = MetadataKey(int)

  #: text_trace_rdy
  #:
  #: Type: ``bool``; input
  #:
  #: Default value: False
  text_trace_rdy = MetadataKey(bool)

  def __init__( self, default_trace_len=8 ):
    self.default_trace_len = default_trace_len

  def __call__( self, top ):

    # Turn on by default
    if top.has_metadata( self.enable ) and top.get_metadata( self.enable ) is False:
      return

    assert not top.has_metadata( self.clear_cl_trace_func )

    top.set_metadata( self.clear_cl_trace_func, self.process_component( top ) )

  @staticmethod
  def _compile_globs( patterns ):
    if patterns is None:
      return []
    return [ re.compile( fnmatch.translate( p ) ) for p in patterns ]

  @staticmethod
  def _match_path( path, include, exclude ):
    inc = any( r.match( path ) for r in include ) if include else True
    if not inc:
      return False
    return not any( r.match( path ) for r in exclude ) if exclude else True

  def process_component( self, top ):

    # Load text-trace configuration from metadata. All text tracing is
    # disabled by default to avoid overhead and noise.
    cfg_text_trace = top.get_metadata( self.text_trace ) \
                     if top.has_metadata( self.text_trace ) else False
    cfg_include = top.get_metadata( self.text_trace_include ) \
                  if top.has_metadata( self.text_trace_include ) else [ "*" ]
    cfg_exclude = top.get_metadata( self.text_trace_exclude ) \
                  if top.has_metadata( self.text_trace_exclude ) else []
    cfg_output = top.get_metadata( self.text_trace_output ) \
                 if top.has_metadata( self.text_trace_output ) else sys.stderr.write
    cfg_max_len = top.get_metadata( self.text_trace_max_len ) \
                  if top.has_metadata( self.text_trace_max_len ) else 120
    cfg_trace_rdy = top.get_metadata( self.text_trace_rdy ) \
                    if top.has_metadata( self.text_trace_rdy ) else False

    include_res = self._compile_globs( cfg_include )
    exclude_res = self._compile_globs( cfg_exclude )

    def _write( msg ):
      if hasattr( cfg_output, "write" ):
        cfg_output.write( msg )
      else:
        cfg_output( msg )

    def _fmt_value( v ):
      s = str( v )
      # A max_len of None or 0 disables truncation so full arguments are
      # available for debug (e.g. large BackwardCtrlBus / IQDispatchReq).
      if cfg_max_len and len( s ) > cfg_max_len:
        s = s[:cfg_max_len] + "..."
      return s

    def _emit_text_trace( ifc, kind, args, kwargs, ret ):
      path = ifc._dsl.full_name
      if not self._match_path( path, include_res, exclude_res ):
        return

      try:
        cycle = top._sim.simulated_cycles
      except AttributeError:
        cycle = -1

      arg_strs = [ _fmt_value( a ) for a in args ] + \
                 [ f"{k}={_fmt_value(v)}" for k, v in kwargs.items() ]
      args_str = "" if not arg_strs else f"({','.join(arg_strs)})"
      ret_str = "" if ret is None else f" -> {_fmt_value(ret)}"

      _write( f"cyc={cycle:4d} {kind:4s} {path}{args_str}{ret_str}\n" )

    # [wrap_callee_method] wraps the original method in a callee port
    # into a new method that not only calls the origianl method, but
    # also saves the arguments to the method and the return value,
    # which can be used for composing the line trace.
    # The wrapped method also need to update the saved arguments and
    # return value of all the methods this callee port is driving.
    def wrap_callee_method( mport, net ):
      mport.raw_method = mport.method
      def wrapped_method( self, *args, **kwargs ):
        # If it has greenlet i.e. blocking ... we need to make sure
        # we record everything after the method is successfully invoked
        ret = self.raw_method( *args, **kwargs )
        for m in net:
          m.called = True
          m.saved_args = args
          m.saved_kwargs = kwargs
          m.saved_ret = ret

        # Emit text trace event if enabled. The parent of a method port
        # inside a CL interface is the interface itself.
        if cfg_text_trace:
          parent = getattr( self._dsl, "parent_obj", None )
          if isinstance( parent, ( NonBlockingIfc, BlockingIfc ) ):
            is_method = ( parent.method is self )
            is_rdy    = getattr( parent, "rdy", None ) is self
            if is_method:
              _emit_text_trace( parent, "CALL", args, kwargs, ret )
            elif cfg_trace_rdy and is_rdy:
              _emit_text_trace( parent, "RDY", args, kwargs, ret )

        return ret
      mport.method = lambda *args, **kwargs : wrapped_method( mport, *args, **kwargs )

    # [wrap_caller_method] wraps the original method in a caller port
    # into a new method that calls its driver instead of the actual
    # method, which will trigger the actual driver to update all other
    # method ports connected to this net.
    def wrap_caller_method( mport, driver_method ):
      def wrapped_method( self, *args, **kwargs ):
        return driver_method( *args, **kwargs )
      mport.method = lambda *args, **kwargs : wrapped_method( mport, *args, **kwargs )

    # Collect all method ports and add some stamps
    all_callees = set()
    all_method_ports = top.get_all_object_filter(
      lambda s: isinstance( s, MethodPort )
    )
    for mport in all_method_ports:
      mport.called = False
      mport.saved_args = None
      mport.saved_kwargs = None
      mport.saved_ret = None
      if isinstance( mport, CalleePort ):
        all_callees.add( mport )

    # Collect all method nets and wrap the actual driving method
    all_drivers = set()
    all_method_nets = top.get_all_method_nets()
    for driver, net in all_method_nets:
      if driver is not None:
        wrap_callee_method( driver, net )
        all_drivers.add( driver )
      for member in net:
        if isinstance( member, CallerPort ):
          assert member is not driver
          wrap_caller_method( member, driver )

    # Handle other callee that is not driving anything
    for mport in ( all_callees - all_drivers ):
      wrap_callee_method( mport, set() )

    # [mk_new_str] replaces [_str_hook] in a non-blocking interface with
    # a new to-string function that uses the metadata to compose line
    # trace.
    # When the rdy is called and returns true, and the method gets called,
    # the line trace just prints out the actual message. Otherwise, it '
    # prints out some special characters under different circumstances:
    # - 'x' rdy not called but method called
    # - '.' rdy not called, method not called
    # - "#" rdy called and is false, method not called
    # - "X" rdy called and is false, method still called
    # - " " rdy called and is true, method not called
    # For example, a cycle-level single element normal queue would have
    # the following line trace:
    #      enq     deq
    #  1:( 0000 () #    ) - enq(0000) called, deq is not ready
    #  2:( #    () 0000 ) - enq is not ready, deq() gets called
    #  3:( 0001 () #    ) - enq(0001) called, deq is not ready again

    def mk_new_str_non_blocking( ifc ):
      def new_str():
        # If rdy is called
        if ifc.rdy.called:
          # If rdy is called and returns true
          if ifc.rdy.saved_ret:
            # If rdy and method called - return actual message
            if ifc.method.called:
              args_strs = [ str( arg ) for arg in ifc.method.saved_args ] + \
                          [ str( arg ) for _, arg in ifc.method.saved_kwargs.items() ]

              ret_str = "" if ifc.method.saved_ret is None else str( ifc.method.saved_ret )

              trace = ""
              if args_strs:
                trace += f"({','.join(args_strs)})"
              if ret_str:
                trace += f"={ret_str}"

              ifc.trace_len = len(trace)
              return trace

            # If rdy and method not called
            else:
              return " ".ljust( ifc.trace_len )

          # If rdy is called and returns false
          elif ifc.method.called:
            return "X".ljust( ifc.trace_len )
          else:
            return "#".ljust( ifc.trace_len )

        # If rdy is not called
        elif ifc.method.called:
          return "x".ljust( ifc.trace_len )

        else:
          return ".".ljust( ifc.trace_len )
      return new_str

    # Collecting all non blocking interfaces and replace the str hook
    for ifc in top.get_all_object_filter( lambda s: isinstance( s, NonBlockingIfc ) ):
      if ifc.method.Type is not None:
        ifc.trace_len = len( str( ifc.method.Type() ) )
      else:
        ifc.trace_len = self.default_trace_len
      ifc._str_hook = mk_new_str_non_blocking( ifc )

    # [mk_new_str] replaces [_str_hook] in a blocking interface with
    # a new to-string function that uses the metadata to compose line
    # trace. The case for blocking interfaces is simpler than
    # non-blocking interfaces as we only have two possibilities
    # called/not called.
    # - " " method not called
    # - msg method called
    def mk_new_str_blocking( ifc ):
      def new_str():
        # If method called - return actual message
        if ifc.method.called:
          args_strs = [ str( arg ) for arg in ifc.method.saved_args ] + \
                      [ str( arg ) for _, arg in ifc.method.saved_kwargs.items() ]

          ret_str = "" if ifc.method.saved_ret is None else str( ifc.method.saved_ret )

          trace = ""
          if args_strs:
            trace += f"({','.join(args_strs)})"
          if ret_str:
            trace += f"={ret_str}"

          ifc.trace_len = len(trace)
          return trace

        # If method not called
        else:
          return " ".ljust( ifc.trace_len )
      return new_str

    # Collecting all blocking interfaces and replace the str hook
    for ifc in top.get_all_object_filter( lambda s: isinstance( s, BlockingIfc ) ):
      if ifc.method.Type is not None:
        ifc.trace_len = len( str( ifc.method.Type() ) )
      else:
        ifc.trace_len = self.default_trace_len
      ifc._str_hook = mk_new_str_blocking( ifc )

    # An update block that resets all method ports to not called
    def reset_method_ports():
      for mport in all_method_ports:
        mport.called = False
        mport.saved_args = None
        mport.saved_kwargs = None
        mport.saved_ret = None

    return reset_method_ports
