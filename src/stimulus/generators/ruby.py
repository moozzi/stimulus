"""Code generators for a Ruby interface of igraph."""

from .base import SingleBlockCodeGenerator
from .utils import remove_prefix

from typing import IO, List, Callable, Sequence, Set, Dict, Optional, Iterator, Tuple
from functools import lru_cache
from dataclasses import dataclass
from contextlib import contextmanager, ExitStack

from stimulus.errors import CodeGenerationError
from stimulus.model.functions import FunctionDescriptor
from stimulus.model.parameters import ParamSpec, DefaultValueType, ParamMode
from stimulus.model.types import TypeDescriptor

@lru_cache(maxsize=128)
def _get_ffi_arg_type_from_c_arg_type(c_type: str):
  # Strip "const" from the fron
  c_type = c_type.strip()
  while c_type.startswith("const "):
    c_type = c_type[6:].strip()

  # Replace pointer asterisks
  wrap_counter = 0
  while c_type.endswith("*"):
    c_type = c_type[:-1].strip()
    wrap_counter += 1

  if c_type in (
    "char",
    "int",
    "float",
    "double",
    "size_t",
    "ssize_t",
    "bool",
    "void",
  ):
    c_type = f":{c_type}"

  while wrap_counter > 0 and c_type == "wchar":
    wrap_counter -= 1
    c_type = ":string"

  while wrap_counter > 0:
    c_type = ":pointer"
    wrap_counter -= 1

  if c_type.startswith("igraph_") or c_type.startswith("handle_igraph_"):
    c_type = c_type.upper()

  return c_type

def _get_ruby_type_from_type_spec(
  type_spec: TypeDescriptor, out: bool = False
) -> Optional[str]:
  if out and "RB_RETURN_TYPE" in type_spec:
    return type_spec.get("RB_RETURN_TYPE")
  elif "RB_TYPE" in type_spec:
    return type_spec.get("RB_TYPE")
  else:
    raise CodeGenerationError(f"no Ruby type known for type: {type_spec.name}")

def _get_ruby_type_from_default_value(default_value: str) -> str:
  match(default_value):
    case 'True':
      return 'true'
    case 'False':
      return 'false'
    case _:
      return default_value

class IndentedWriter:
    """Helper class to dynamically manage indentation levels while creating the
    body of a function.
    """

    _indentation: str = "  "
    """The indentation prefix for each line for a single indentation level."""

    _level: int = 0
    """The current indentation level."""

    _writer: Callable[[str], None]
    """The writer function to wrap."""

    def __init__(self, writer: Callable[[str], None], *, level: int = 0):
        """Constructor."""
        self._writer = writer
        self._level = level

    @contextmanager
    def indent(self) -> Iterator[None]:
        """Context manager that increases the current indentation level
        while the execution is in the context.
        """
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    def write(self, line: str) -> None:
        if line:
            line = (self._indentation * self._level) + line
        self._writer(line)

    __call__ = write

class RubyFFICodeGenerator(SingleBlockCodeGenerator):
  bitfield_types: Set[str]
  enum_type: Set[str]
  lines: List[str]

  def generate_preamble(self, inputs: Sequence[str], output: IO[str]) -> None:
    self.bitfield_types = set()
    self.enum_types = set()
    self.lines = []
    return super().generate_preamble(inputs, output)

  def generate_function(self, name: str, out: IO[str]) -> None:
    self.lines.append("")
    try:
      self._generate_function(name, self.lines.append)
    except CodeGenerationError as ex:
      self.lines.append(f"# {name}: {ex}")

  def _generate_function(self, name: str, write: Callable[[str], None]) -> None:
    # Check types
    self.check_types_of_function(name)

    # Get function specification
    spec = self.get_function_descriptor(name)

    # Construct Ruby return type
    return_type = self.get_type_descriptor(spec.return_type)
    rb_return_type: Optional[str] = return_type.get("FFI_RETURN_TYPE")
    if not rb_return_type:
      # Try deriving the ffi type
      rb_return_type = _get_ffi_arg_type_from_c_arg_type(
        return_type.get_c_type()
      )

    if rb_return_type.startswith("igraph_") or rb_return_type.startswith("handle_igraph"):
      rb_return_type = rb_return_type.upper()

    if return_type.is_enum:
      self.enum_types.add(rb_return_type)
    if return_type.is_bitfield:
      self.bitfield_types.add(rb_return_type)

    rb_arg_types: List[str] = []
    for parameter in spec.iter_parameters():
      if parameter.is_deprecated:
        continue

      param_type = self.get_type_descriptor(parameter.type)
      c_arg_type = param_type.declare_c_function_argument(mode=parameter.mode)
      if not c_arg_type:
        # This argument is not present in the C function calls
        continue

      rb_arg_type = _get_ffi_arg_type_from_c_arg_type(c_arg_type) # Change to ruby types later
      rb_arg_types.append(rb_arg_type)

      if param_type.is_enum:
        self.enum_types.add(rb_arg_type)
      if param_type.is_bitfield:
        self.bitfield_types.add(rb_arg_type)

    rb_arg_types_joined = ", ".join(rb_arg_types)
    write(f"  attach_function :{name}, [{rb_arg_types_joined}], {rb_return_type}")

  def generate_epilogue(self, inputs: Sequence[str], output: IO[str]) -> None:
      write = output.write

      if self.enum_types:
          write("# Set up aliases for all enum types\n")
          write("\n")
          for enum_type in sorted(self.enum_types):
            if enum_type != ":pointer":
                  write(f"{enum_type} = :int\n")
          write("\n")

      if self.bitfield_types:
          write("# Set up aliases for all bitfield types\n")
          write("\n")
          for bitfield_type in sorted(self.bitfield_types):
              write(f"{bitfield_type} = :int\n")
          write("\n")

      write("# Add argument and return types for functions imported from igraph\n")
      write("module IgraphC")
      write("\n  extend FFI::Library")
      write("\n\n  ffi_lib :igraph\n")
      write("\n".join(self.lines))
      write("\nend\n")

      return super().generate_epilogue(inputs, output)

@dataclass
class ArgInfo:
  param_spec: ParamSpec
  type_spec: TypeDescriptor

  c_name: str
  rb_name: str
  rb_type: str

  appears_in_argument_list: bool = False
  default_value: Optional[str] = None

  @classmethod
  def from_param_spec(
    cls, spec: ParamSpec, type_descriptor_getter: Callable[[str], TypeDescriptor]
  ):
    type = type_descriptor_getter(spec.type)

    rb_name = spec.name.lower()

    # Translate Ruby reserved keywords
    if rb_name in ("in"):
      rb_name += "_"

    c_name = f"c_{spec.name}"

    rb_type = _get_ruby_type_from_type_spec(type)
    if spec.is_optional and rb_type:
      rb_type = "Optional"

    result = cls(
      param_spec=spec,
      type_spec=type,
      c_name=c_name,
      rb_name=rb_name,
      rb_type=rb_type or "nil",
    )

    if rb_type is None:
      result.appears_in_argument_list = False
    elif spec.is_deprecated:
      result.appears_in_argument_list = False
    else:
      result.appears_in_argument_list = spec.is_input

    if spec.has_default_value:
      default_value = spec.get_default_value(type) or "nil"
      result.default_value = _get_ruby_type_from_default_value(default_value)

      if result.default_value == "nil":
        result.default_value = "nil"

      if (
        type.is_enum
        and spec.default is not None
        and spec.default[0] == DefaultValueType.ABSTRACT
        and result.default_value == spec.default[1]
        and rb_type is not None
      ):
        result.default_value = rb_type + "::" + result.default_value
    else:
      result.default_value = None

    return result

  def get_ruby_declaration(self) -> str:
      """Returns the declaration of this argument for the Ruby function header."""
      if self.default_value is None and not self.param_spec.is_optional:
          return f"{self.rb_name}"
      elif self.default_value is None:
          return f"{self.rb_name} = nil"
      else:
          return f"{self.rb_name} = {self.default_value}"

  def get_argument_for_function_call(self, args: Dict[str, "ArgInfo"]) -> str:
    template = self.type_spec.get("CALL")
    if template:
      return self._apply_replacements(template, args)
    else:
      return self.c_name

  def get_input_conversion(self, args: Dict[str, "ArgInfo"]) -> Optional[str]:
      if not self.appears_in_argument_list:
          default = "%C% = nil"
      elif self.param_spec.is_input:
          default = "%C% = %I%"
      else:
          default = ""

      template = self.type_spec.get_input_conversion_template_for(
        self.param_spec.mode, default=default
      )
      if not template:
          if not self.param_spec.is_input:
              raise CodeGenerationError(
                  f"Cannot construct an instance of abstract type {self.type_spec.name}"
              )
          else:
              return None
      else:
          if (
            self.param_spec.is_input
            and self.param_spec.is_optional
            and self.default_value in (None, "None")
          ):
              template = f"{template} if %I%"

          return self._apply_replacements(template, args)

  def get_output_conversion(self, args: Dict[str, "ArgInfo"]) -> Optional[str]:
    if self.param_spec.mode == ParamMode.OUT:
      default = "%I% = %C%.value"
    else:
      default = ""
    template = self.type_spec.get_output_conversion_template_for(
      self.param_spec.mode, default=default
    )

    if not template:
      return None
    else:
      return self._apply_replacements(template, args)

  @property
  def needs_exit_stack(self) -> bool:
    """Returns whether this argument needs an exit stack for properly
    handling its input and/or output conversions.
    """
    return self.type_spec.has_flag("stack")

  def _apply_replacements(self, value: str, args: Dict[str, "ArgInfo"]) -> str:
    value = value.replace("%I%", self.rb_name)
    value = value.replace("%C%", self.c_name)
    if self.needs_exit_stack:
      value = value.replace("%S%", "rb__stack")
    # TODO: EXIT STACK?

    for index, dep in enumerate(self.param_spec.dependencies, 1):
        arg = args.get(dep)
        if arg is None:
            raise CodeGenerationError(
                f"Unknown dependency for parameter {self.rb_name!r}: {dep!r}"
            )
        value = value.replace(f"%I{index}%", arg.rb_name)
        value = value.replace(f"%C{index}%", arg.c_name)

    return value

class RubyRubyCodeGenerator(SingleBlockCodeGenerator):
  def generate_function(self, name: str, output: IO[str]) -> None:
    write = output.write

    lines = [""]
    try:
      self._generate_function(name, lines.append)
      lines.append("")
      write("\n".join(lines))
    except CodeGenerationError as ex:
      write(f"\n# {name}: {ex}\n")

  def _generate_function(self, name: str, write: Callable[[str], None]) -> None:
    writer = IndentedWriter(write)
    write = writer.write

    # Check types
    self.check_types_of_function(name)

    # Get function specification
    spec = self.get_function_descriptor(name)

    # Derive Ruby name of the function from its C name
    rb_name = self._get_ruby_name(spec)

    # Construct Ruby arguments
    args = self._process_argument_list(spec)

    arg_specs = [
      arg_spec
      for arg_spec in spec.iter_reordered_parameters()
      if args[arg_spec.name].appears_in_argument_list
    ]
    arg_specs = sorted(
      arg_specs,
      key=lambda arg_spec: 1
      if arg_spec.default is None and not arg_spec.is_optional
      else 2,
    )

    return_arg_names, return_types = self._get_return_args(spec)

    rb_args = [
      args[arg_spec.name].get_ruby_declaration() for arg_spec in arg_specs
    ]

    # TODO: ADD DOCUMENTATION
    write(f"def {rb_name}({', '.join(rb_args)})")

    with ExitStack() as stack:
        stack.enter_context(writer.indent())

        convs = [
          args[param_spec.name].get_input_conversion(args)
          for param_spec in spec.iter_parameters()
        ] 
        if convs:
            write("# Prepare input arguments")
            for conv in convs:
                write(conv)
            write("")

        write("# Call wrapped function")
        needs_return_value_from_c_call = "" in return_arg_names
        c_args = ", ".join(
          args[arg_spec.name].get_argument_for_function_call(args)
          for arg_spec in spec.iter_parameters()
        )
        c_call = f"IgraphC.{name}({c_args})"
        if needs_return_value_from_c_call:
            c_call = f"c__result = {c_call}"
        write(c_call)

        # Add output conversion calls
        convs = [
          args[param_spec.name].get_output_conversion(args)
          for param_spec in spec.iter_parameters()
        ]
        convs = [conv for conv in convs if conv]
        if convs:
          write("")
          write("# Prepare output arguments")
          for conv in convs:
            write(conv)

        if return_arg_names:
          return_var = "c__result"

          try:
            idx = return_arg_names.index("")
          except ValueError:
            tmpl = ""
            idx = -1
          else:
            tmpl = return_types[idx].get_output_conversion_template_for(
              ParamMode.OUT
            )

          if tmpl:
            conv = tmpl.replace("%I%", "rb__result").replace("%C%", "c__result")
            return_var = "rb__result"
            write("")
            write("# Prepare return value")
            write(conv)

          write("")
          write("# Construct return value")
          if len(return_arg_names) == 1:
            if needs_return_value_from_c_call:
              var_name = return_var
            else:
              var_name = args[return_arg_names[0]].rb_name
            write(var_name)
          else:
            joint_parts = ", ".join(
              args[name].rb_name if name else return_var
              for name in return_arg_names
            )
            write(f"return {joint_parts}")

    write("end")

  def _get_ruby_name(self, spec: FunctionDescriptor) -> str:
    return spec.get("NAME") or remove_prefix(spec.name, "igraph_")

  def _process_argument_list(self, spec: FunctionDescriptor) -> Dict[str, ArgInfo]:
    return {
      param.name: ArgInfo.from_param_spec(param, self.get_type_descriptor)
      for param in spec.iter_parameters()
    }

  def _get_return_args(self, spec: FunctionDescriptor) -> Tuple[List[str], List[TypeDescriptor]]:
    arg_names: List[str] = []
    arg_types: List[TypeDescriptor] = []

    return_type = self.get_type_descriptor(spec.return_type)
    if return_type.name != "ERROR" and return_type.name != "VOID":
        arg_names.append("")
        arg_types.append(return_type)

    for parameter in spec.iter_parameters():
      if not parameter.is_deprecated and not parameter.is_input:
          arg_names.append(parameter.name)
          arg_types.append(self.get_type_descriptor(parameter.type))

    return arg_names, arg_types
