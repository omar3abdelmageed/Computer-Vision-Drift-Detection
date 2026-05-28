from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.common import ValidationIssue, ValidationStatus, YamlCandidate


def discover_yamls(dataset_root: Path) -> list[YamlCandidate]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for detection dataset validation.") from exc

    candidates: list[YamlCandidate] = []
    paths = sorted([*dataset_root.rglob("*.yaml"), *dataset_root.rglob("*.yml")])
    for path in paths:
        issues: list[ValidationIssue] = []
        parsed: dict[str, Any] = {}
        is_valid = False
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(content, dict):
                parsed = content
                is_valid = {"train", "val", "names"}.issubset(parsed.keys())
                if not is_valid:
                    issues.append(ValidationIssue(ValidationStatus.WARNING, "yaml_missing_detection_keys", "YAML does not contain train, val, and names.", str(path)))
            else:
                issues.append(ValidationIssue(ValidationStatus.INVALID, "yaml_not_mapping", "YAML content is not a mapping.", str(path)))
        except Exception as exc:
            issues.append(ValidationIssue(ValidationStatus.INVALID, "yaml_parse_error", str(exc), str(path)))
        candidates.append(YamlCandidate(path=path, filename=path.name, is_valid=is_valid, parsed_content=parsed, issues=issues))
    return sorted(candidates, key=lambda c: (c.path.parent != dataset_root, c.filename != "data.yaml", str(c.path)))


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []
