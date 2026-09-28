# SPDX-FileCopyrightText: 2026 Open Source Robotics Foundation, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Semantic validation helpers for ROS parameter definitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


def validate_parameter_names(parameters: Iterable[str]) -> str | None:
    """Validate parameter names, returning an error message if validation fails."""
    parameter_names = list(parameters)
    for name in parameter_names:
        if any(not part for part in name.split('.')):
            return f'invalid dotted parameter name {name!r}: name components cannot be empty'

    conflict = find_parameter_namespace_conflict(parameter_names)
    if conflict is not None:
        parameter, nested_parameter = conflict
        return f'parameter {parameter!r} conflicts with {nested_parameter!r}: a parameter cannot also be a parameter namespace'

    return None


def find_parameter_namespace_conflict(names: Iterable[str]) -> tuple[str, str] | None:
    """Return the first parameter pair where one name is the other's namespace.

    Parameters with a common namespace are valid, such as ``colour.r`` and
    ``colour.g``.
    A parameter cannot also be a namespace, so ``colour`` and ``colour.r``
    conflict.
    """
    names_set = set(names)
    for name in sorted(names_set):
        parts = name.split('.')
        for part_count in range(1, len(parts)):
            namespace = '.'.join(parts[:part_count])
            if namespace in names_set:
                return namespace, name
    return None


# Validators grouped by the parameter types they apply to.
# Names are the base name, without the optional ``<>`` suffix.
_NUMERIC_COMPARISON_VALIDATORS = frozenset({'bounds', 'lt', 'gt', 'lt_eq', 'gt_eq'})
_SCALAR_VALIDATORS = frozenset({'one_of'})
_SIZE_VALIDATORS = frozenset({'fixed_size', 'size_gt', 'size_lt', 'not_empty'})
_ARRAY_VALIDATORS = frozenset({'unique', 'subset_of'})
_NUMERIC_ARRAY_VALIDATORS = frozenset({'element_bounds', 'lower_element_bounds', 'upper_element_bounds'})
_NUMERIC_BASE_TYPES = frozenset({'int', 'double', 'byte'})
_RANGE_VALIDATORS = frozenset({'bounds', 'element_bounds'})


@dataclass(frozen=True)
class ParameterType:
    """A parameter ``type`` string split into its parts."""

    base: str
    is_array: bool
    fixed_size: int | None

    @classmethod
    def parse(cls, type_name: str) -> ParameterType:
        """Parse a schema-valid parameter type, such as ``double_array_fixed_3``."""
        name, fixed_size = type_name, None
        if '_fixed_' in type_name:
            name, size = type_name.rsplit('_fixed_', 1)
            fixed_size = int(size)
        base, is_array = (name.removesuffix('_array'), True) if name.endswith('_array') else (name, False)
        return cls(base=base, is_array=is_array, fixed_size=fixed_size)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _matches_base_type(value: Any, base: str) -> bool:
    """Return whether a single value is valid for a parameter base type."""
    if base == 'bool':
        return isinstance(value, bool)
    if base == 'int':
        return _is_int(value)
    if base == 'double':
        return _is_int(value) or isinstance(value, float)
    if base == 'string':
        return isinstance(value, str)
    if base == 'byte':
        return _is_int(value) and 0 <= value <= 255
    return False


def _check_default_value(type_name: str, default: Any) -> str | None:
    """Check that a default value matches its declared parameter type."""
    param_type = ParameterType.parse(type_name)
    if param_type.base == 'none':
        return f'type {type_name!r} does not take a default_value'

    if param_type.is_array:
        if not isinstance(default, list):
            return f'default_value {default!r} is not a list, as type {type_name!r} requires'
        for element in default:
            if not _matches_base_type(element, param_type.base):
                return f'default_value element {element!r} does not match type {type_name!r}'
    elif not _matches_base_type(default, param_type.base):
        return f'default_value {default!r} does not match type {type_name!r}'

    if param_type.fixed_size is not None and len(default) > param_type.fixed_size:
        return f'default_value has length {len(default)}, more than type {type_name!r} allows'

    return None


def _validator_applies(validator: str, param_type: ParameterType) -> bool:
    """Return whether a built-in validator applies to a parameter type."""
    if param_type.base == 'none':
        return False
    numeric = param_type.base in _NUMERIC_BASE_TYPES
    if validator in _NUMERIC_COMPARISON_VALIDATORS:
        return numeric and not param_type.is_array
    if validator in _SCALAR_VALIDATORS:
        return not param_type.is_array
    if validator in _SIZE_VALIDATORS:
        return param_type.is_array or param_type.base == 'string'
    if validator in _ARRAY_VALIDATORS:
        return param_type.is_array
    if validator in _NUMERIC_ARRAY_VALIDATORS:
        return numeric and param_type.is_array
    return False


def _as_list(arguments: Any) -> list:
    if arguments is None:
        return []
    return arguments if isinstance(arguments, list) else [arguments]


def _validator_values(validator: str, arguments: Any) -> list:
    """Return the values a validator compares the parameter (or its elements) against."""
    if validator in _SCALAR_VALIDATORS | _ARRAY_VALIDATORS:
        return [value for group in _as_list(arguments) for value in group]
    if validator in _NUMERIC_COMPARISON_VALIDATORS | _NUMERIC_ARRAY_VALIDATORS:
        return _as_list(arguments)
    # Size validators take lengths, not parameter values.
    return []


def _check_validator(type_name: str, validator_name: str, arguments: Any) -> str | None:
    """Check that a built-in validator applies to the parameter type, with matching arguments."""
    param_type = ParameterType.parse(type_name)
    validator = validator_name.removesuffix('<>')
    if not _validator_applies(validator, param_type):
        return f'validator {validator_name!r} does not apply to type {type_name!r}'

    for value in _validator_values(validator, arguments):
        if not _matches_base_type(value, param_type.base):
            return f'validator {validator_name!r} argument {value!r} does not match type {type_name!r}'

    if validator in _RANGE_VALIDATORS:
        lower, upper = arguments
        if lower > upper:
            return f'validator {validator_name!r} lower bound {lower!r} is greater than upper bound {upper!r}'

    return None


def _is_custom_validator(validator_name: str) -> bool:
    return '::' in validator_name


def check_parameter_definition(definition: Mapping[str, Any]) -> str | None:
    """Check one schema-valid parameter definition, returning an error message if it is invalid.

    Covers what JSON Schema cannot express:
    the ``default_value`` must match ``type``,
    and each built-in validator must apply to ``type`` with arguments of the parameter's element type.
    Custom (namespace-qualified) validators are not checked.
    """
    type_name = definition['type']
    if 'default_value' in definition and (error := _check_default_value(type_name, definition['default_value'])):
        return error

    for validator_name, arguments in (definition.get('validation') or {}).items():
        if _is_custom_validator(validator_name):
            continue
        if error := _check_validator(type_name, validator_name, arguments):
            return error

    return None


def validate_parameter_definitions(parameters: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Validate each parameter definition, returning an error message for the first invalid one."""
    for name, definition in parameters.items():
        if error := check_parameter_definition(definition):
            return f'parameter {name!r}: {error}'
    return None
