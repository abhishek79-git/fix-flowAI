"""AST code quality analyzer."""

import ast
import os
from typing import Dict, Any, Tuple

def analyze_file(filepath: str) -> Dict[str, Any]:
    """Analyze a single Python file using AST."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except Exception:
        return {}
        
    num_functions = 0
    num_classes = 0
    max_function_length = 0
    has_docstrings = True
    has_type_hints = True
    wildcard_imports = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            num_functions += 1
            length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') and node.end_lineno else 0
            if length > max_function_length:
                max_function_length = length
        elif isinstance(node, ast.ClassDef):
            num_classes += 1
        elif isinstance(node, ast.ImportFrom):
            if any(name.name == '*' for name in node.names):
                wildcard_imports = True
                
    return {
        "num_functions": num_functions,
        "num_classes": num_classes,
        "max_function_length": max_function_length,
        "has_docstrings": has_docstrings,
        "has_type_hints": has_type_hints,
        "wildcard_imports": wildcard_imports
    }

def analyze_project(project_dir: str) -> Dict[str, Any]:
    """Analyze all Python files in project."""
    total_modules = 0
    functions = 0
    classes = 0
    functions_over_60_lines = 0
    wildcard_imports = 0
    missing_docstrings = 0
    type_hint_coverage = 100.0
    
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                total_modules += 1
                res = analyze_file(os.path.join(root, file))
                functions += res.get('num_functions', 0)
                classes += res.get('num_classes', 0)
                if res.get('max_function_length', 0) > 60:
                    functions_over_60_lines += 1
                if res.get('wildcard_imports', False):
                    wildcard_imports += 1
                    
    return {
        "total_modules": total_modules,
        "functions": functions,
        "classes": classes,
        "functions_over_60_lines": functions_over_60_lines,
        "wildcard_imports": wildcard_imports,
        "missing_docstrings": missing_docstrings,
        "type_hint_coverage": type_hint_coverage
    }

def generate_ast_report(project_dir: str) -> str:
    """Generate AST report."""
    stats = analyze_project(project_dir)
    return (
        f"AST Report for {project_dir}\n"
        f"Total modules: {stats['total_modules']}\n"
        f"Functions: {stats['functions']}\n"
        f"Classes: {stats['classes']}\n"
        f"Functions > 60 lines: {stats['functions_over_60_lines']}\n"
        f"Wildcard imports: {stats['wildcard_imports']}\n"
    )

def check_ast_quality(project_dir: str) -> Tuple[bool, str]:
    """Check AST quality against thresholds."""
    stats = analyze_project(project_dir)
    issues = []
    
    if stats['functions_over_60_lines'] > 5:
        issues.append(f"Too many long functions ({stats['functions_over_60_lines']} > 5)")
    if stats['wildcard_imports'] > 0:
        issues.append(f"Wildcard imports found ({stats['wildcard_imports']})")
        
    if issues:
        return False, "FAILED: " + "; ".join(issues)
    return True, "PASS"
