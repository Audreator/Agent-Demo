from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    pass


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise SchemaValidationError("参数必须是 JSON 对象")
    required = schema.get("required") or []
    for key in required:
        if key not in arguments or arguments[key] is None:
            raise SchemaValidationError(f"缺少必填参数: {key}")
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        extra = set(arguments) - set(properties)
        if extra:
            raise SchemaValidationError(f"包含未定义参数: {', '.join(sorted(extra))}")
    for key, value in arguments.items():
        if key not in properties or value is None:
            continue
        rule = properties[key]
        expected = rule.get("type")
        if expected:
            python_type: Any = _TYPE_MAP.get(expected)
            if python_type is None:
                continue
            if expected in {"integer", "number"} and isinstance(value, bool):
                raise SchemaValidationError(f"参数 {key} 必须是 {expected}")
            if expected == "number" and isinstance(value, int):
                pass
            elif not isinstance(value, python_type):
                raise SchemaValidationError(f"参数 {key} 必须是 {expected}")
        if "enum" in rule and value not in rule["enum"]:
            raise SchemaValidationError(f"参数 {key} 必须是 {rule['enum']} 之一")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                raise SchemaValidationError(f"参数 {key} 不能小于 {rule['minimum']}")
            if "maximum" in rule and value > rule["maximum"]:
                raise SchemaValidationError(f"参数 {key} 不能大于 {rule['maximum']}")
        if expected == "string" and isinstance(value, str):
            if "maxLength" in rule and len(value) > rule["maxLength"]:
                raise SchemaValidationError(f"参数 {key} 长度超限")


__all__ = ["SchemaValidationError", "validate_arguments"]
