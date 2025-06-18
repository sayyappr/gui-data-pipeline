# validator.py or utils/logger.py

from collections import defaultdict
import json
import os
import ast
import warnings
import pyautogui

# Collector dictionary
element_type_summary = defaultdict(lambda: {'count': 0, 'instructions': []})

def update_element_type_info(element_type: str, instruction: str):
    element_type_summary[element_type]['count'] += 1
    element_type_summary[element_type]['instructions'].append(instruction.strip())

def dump_element_type_info(output_path="logs/element_type_summary.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(element_type_summary, f, indent=2)

def validate_pyautogui_command(cmd: str) -> bool:
    try:
        tree = ast.parse(cmd, mode='exec')
    except SyntaxError:
        warnings.warn(f"[pyautogui] Syntax error: {cmd}")
        return False

    # Check if it's a single expression like pyautogui.click(x=0.5, y=0.5)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        warnings.warn(f"[pyautogui] Not a single expression: {cmd}")
        return False

    expr = tree.body[0].value
    if not isinstance(expr, ast.Call):
        warnings.warn(f"[pyautogui] Not a function call: {cmd}")
        return False

    # Check function is pyautogui.<method>
    if not (isinstance(expr.func, ast.Attribute) and
            isinstance(expr.func.value, ast.Name) and
            expr.func.value.id == 'pyautogui'):
        warnings.warn(f"[pyautogui] Not a pyautogui method: {cmd}")
        return False

    method_name = expr.func.attr
    if not hasattr(pyautogui, method_name):
        warnings.warn(f"[pyautogui] Unknown method: {method_name} in {cmd}")
        return False

    return True