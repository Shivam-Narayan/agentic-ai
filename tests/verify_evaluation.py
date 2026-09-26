#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verification script for the enterprise evaluation and monitoring system.
Runs all checks to ensure the system is properly configured.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and is readable."""
    if not path.exists():
        print(f"❌ {description}: File not found at {path}")
        return False
    try:
        if path.suffix == ".json":
            json.loads(path.read_text())
        elif path.suffix in (".py", ".md"):
            path.read_text(encoding="utf-8")
        print(f"✅ {description}: OK")
        return True
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False


def check_env_variables() -> bool:
    """Check if LangSmith environment variables are documented."""
    env_example = ROOT / ".env.example"
    content = env_example.read_text()
    required = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
    missing = [var for var in required if var not in content]
    if missing:
        print(f"❌ .env.example missing: {', '.join(missing)}")
        return False
    print("✅ .env.example has LangSmith variables")
    return True


def check_evaluation_dataset() -> bool:
    """Validate the evaluation dataset."""
    dataset_path = Path(__file__).parent / "eval_questions.json"
    if not dataset_path.exists():
        print("❌ eval_questions.json not found")
        return False
    
    data = json.loads(dataset_path.read_text())
    if not isinstance(data, list) or len(data) == 0:
        print("❌ eval_questions.json is empty or invalid")
        return False
    
    # Check required fields
    required_fields = ["id", "question", "ground_truth", "datasource"]
    issues = []
    for item in data:
        missing = [f for f in required_fields if f not in item]
        if missing:
            issues.append(f"{item.get('id', 'unknown')}: missing {missing}")
    
    if issues:
        print(f"❌ Evaluation dataset has issues:")
        for issue in issues[:3]:  # Show first 3
            print(f"   - {issue}")
        return False
    
    print(f"✅ Evaluation dataset: {len(data)} questions with all required fields")
    return True


def check_monitoring_config() -> bool:
    """Validate monitoring configuration."""
    config_path = Path(__file__).parent / "monitoring_config.json"
    if not config_path.exists():
        print("❌ monitoring_config.json not found")
        return False
    
    config = json.loads(config_path.read_text())
    required_sections = ["alerting_thresholds", "drift_detection", "langsmith_dashboard"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        print(f"❌ monitoring_config.json missing sections: {', '.join(missing)}")
        return False
    
    print(f"✅ Monitoring config: {len(config['alerting_thresholds'])} threshold categories")
    return True


def check_evaluate_script() -> bool:
    """Verify evaluate.py imports and syntax."""
    try:
        import ast
        eval_path = Path(__file__).parent / "evaluate.py"
        ast.parse(eval_path.read_text(encoding="utf-8"))
        print("✅ evaluate.py: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"❌ evaluate.py: Syntax error - {e}")
        return False
    except Exception as e:
        print(f"❌ evaluate.py: Error - {e}")
        return False


def main():
    print("="*70)
    print("  DataDialogue Evaluation System Verification")
    print("="*70)
    print()
    
    checks = [
        (check_env_variables, "Environment configuration"),
        (check_evaluation_dataset, "Evaluation dataset"),
        (check_monitoring_config, "Monitoring configuration"),
        (check_evaluate_script, "Evaluation script"),
        (lambda: check_file_exists(
            ROOT / "docs" / "MONITORING_GUIDE.md",
            "Monitoring guide"
        ), "Documentation"),
    ]
    
    results = []
    for check_fn, name in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print(f"❌ {name}: Unexpected error - {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("="*70)
    print(f"  SUMMARY: {passed}/{total} checks passed")
    print("="*70)
    
    if passed == total:
        print("\n✅ All checks passed! The evaluation system is ready.")
        print("\nNext steps:")
        print("  1. Set up .env with LangSmith credentials")
        print("  2. Run: python tests/evaluate.py --save-baseline")
        print("  3. Check traces at: https://smith.langchain.com/")
        return 0
    else:
        print("\n❌ Some checks failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
