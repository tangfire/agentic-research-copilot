from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .schemas import ResearchSkill, SkillExecutionResult, SkillScript


@dataclass(slots=True)
class LoadedSkillPack:
    skill: ResearchSkill
    root_path: Path
    manifest_path: Path | None
    readme_path: Path | None
    instructions: str
    scripts_by_name: dict[str, SkillScript]


class SkillRegistry:
    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        fallback_catalog: Sequence[ResearchSkill] = (),
        script_timeout_seconds: float = 10.0,
    ) -> None:
        self.root_paths = [self._resolve_path(root) for root in roots]
        self.script_timeout_seconds = script_timeout_seconds
        self._packs_by_id = self._discover_packs()
        if not self._packs_by_id and fallback_catalog:
            self._packs_by_id = {
                skill.skill_id: LoadedSkillPack(
                    skill=skill,
                    root_path=Path(),
                    manifest_path=None,
                    readme_path=None,
                    instructions=skill.instructions_excerpt,
                    scripts_by_name={script.name: script for script in skill.scripts},
                )
                for skill in fallback_catalog
            }

    def list_skills(self) -> list[ResearchSkill]:
        skills = [pack.skill for pack in self._packs_by_id.values()]
        return sorted(skills, key=lambda item: (not item.metadata.get("default", False), item.name.lower()))

    def get_skill(self, skill_id: str | None) -> ResearchSkill | None:
        if not skill_id:
            return None
        pack = self._packs_by_id.get(skill_id)
        return pack.skill if pack is not None else None

    def get_pack(self, skill_id: str | None) -> LoadedSkillPack | None:
        if not skill_id:
            return None
        return self._packs_by_id.get(skill_id)

    def run_script(
        self,
        skill_id: str,
        script_name: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> SkillExecutionResult:
        pack = self.get_pack(skill_id)
        if pack is None:
            raise KeyError(skill_id)
        script = pack.scripts_by_name.get(script_name)
        if script is None:
            raise KeyError(script_name)
        if not script.enabled:
            raise ValueError(f"Skill script '{script_name}' is disabled")
        script_path = (pack.root_path / script.path).resolve()
        if not script_path.exists():
            raise FileNotFoundError(str(script_path))
        if not script_path.is_file():
            raise ValueError(f"Skill script path is not a file: {script_path}")
        root_path = pack.root_path.resolve()
        if root_path and root_path.exists() and not script_path.is_relative_to(root_path):
            raise ValueError(f"Skill script path escapes skill root: {script_path}")

        started_at = _utc_now()
        timeout = timeout_seconds or script.timeout_seconds or self.script_timeout_seconds
        request_payload = {
            "skill_id": skill_id,
            "skill_name": pack.skill.name,
            "script_name": script_name,
            "skill_root": str(pack.root_path),
            "manifest_path": str(pack.manifest_path) if pack.manifest_path else "",
            "instruction_path": str(pack.readme_path) if pack.readme_path else "",
            "instructions_excerpt": pack.skill.instructions_excerpt,
            "input": payload or {},
        }
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            input=json.dumps(request_payload, ensure_ascii=False, default=str),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(pack.root_path),
            timeout=timeout,
            env={
                **_minimal_env(),
                "ARC_SKILL_ID": skill_id,
                "ARC_SKILL_NAME": pack.skill.name,
                "ARC_SKILL_ROOT": str(pack.root_path),
                "ARC_SKILL_SCRIPT_NAME": script_name,
            },
        )
        finished_at = _utc_now()
        output = _parse_json_output(completed.stdout)
        status = "completed" if completed.returncode == 0 else "failed"
        return SkillExecutionResult(
            skill_id=skill_id,
            script_name=script_name,
            status=status,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            output=output,
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "skill_root": str(pack.root_path),
                "script_path": str(script_path),
                "manifest_path": str(pack.manifest_path) if pack.manifest_path else "",
                "instruction_path": str(pack.readme_path) if pack.readme_path else "",
                "timeout_seconds": timeout,
            },
        )

    def describe_skill(self, skill_id: str) -> dict[str, Any]:
        pack = self.get_pack(skill_id)
        if pack is None:
            raise KeyError(skill_id)
        return {
            "skill": pack.skill.model_dump(mode="json"),
            "skill_root": str(pack.root_path),
            "manifest_path": str(pack.manifest_path) if pack.manifest_path else "",
            "instruction_path": str(pack.readme_path) if pack.readme_path else "",
            "instructions": pack.instructions,
            "scripts": [script.model_dump(mode="json") for script in pack.skill.scripts],
        }

    def _discover_packs(self) -> dict[str, LoadedSkillPack]:
        packs: dict[str, LoadedSkillPack] = {}
        for root in self.root_paths:
            if not root.exists():
                continue
            candidates = [root] if _looks_like_skill_pack(root) else [child for child in root.iterdir() if child.is_dir()]
            for candidate in candidates:
                pack = self._load_pack(candidate)
                if pack is not None:
                    packs[pack.skill.skill_id] = pack
        return packs

    def _load_pack(self, root: Path) -> LoadedSkillPack | None:
        manifest_path = root / "skill.json"
        readme_path = root / "SKILL.md"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                manifest = {"skill_id": root.name, "metadata": {"load_error": str(exc)}}
        instructions = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        skill = self._build_skill(root=root, manifest=manifest, instructions=instructions)
        if skill is None:
            return None
        scripts_by_name = {script.name: script for script in skill.scripts}
        return LoadedSkillPack(
            skill=skill,
            root_path=root.resolve(),
            manifest_path=manifest_path if manifest_path.exists() else None,
            readme_path=readme_path if readme_path.exists() else None,
            instructions=instructions,
            scripts_by_name=scripts_by_name,
        )

    def _build_skill(self, *, root: Path, manifest: dict[str, Any], instructions: str) -> ResearchSkill | None:
        skill_id = str(manifest.get("skill_id") or root.name).strip()
        if not skill_id:
            return None
        name = str(manifest.get("name") or _heading_from_markdown(instructions) or skill_id).strip()
        scenario = str(manifest.get("scenario") or _first_paragraph(instructions) or "Reusable research playbook.").strip()
        trigger_keywords = _as_string_list(manifest.get("trigger_keywords")) or _keywords_from_name(skill_id)
        required_inputs = _as_string_list(manifest.get("required_inputs"))
        plan_template = _as_string_list(manifest.get("plan_template"))
        evaluation_focus = _as_string_list(manifest.get("evaluation_focus"))
        script_entries = manifest.get("scripts") if isinstance(manifest.get("scripts"), list) else []
        scripts = []
        for entry in script_entries:
            if not isinstance(entry, dict):
                continue
            script_path = str(entry.get("path") or "").strip()
            if not script_path:
                continue
            scripts.append(
                SkillScript(
                    name=str(entry.get("name") or Path(script_path).stem),
                    path=script_path,
                    description=str(entry.get("description") or ""),
                    enabled=bool(entry.get("enabled", True)),
                    auto=bool(entry.get("auto", False)),
                    timeout_seconds=float(entry.get("timeout_seconds", self.script_timeout_seconds)),
                    input_schema=entry.get("input_schema") if isinstance(entry.get("input_schema"), dict) else {},
                    output_schema=entry.get("output_schema") if isinstance(entry.get("output_schema"), dict) else {},
                    metadata=entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {},
                )
            )
        instructions_excerpt = str(manifest.get("instructions_excerpt") or _markdown_excerpt(instructions)).strip()
        skill_type = str(manifest.get("skill_type") or "pack")
        version = str(manifest.get("version") or "1.0.0")
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        metadata = {
            **metadata,
            "skill_root": str(root.resolve()),
            "manifest_path": str(root / "skill.json") if (root / "skill.json").exists() else "",
            "instruction_path": str(root / "SKILL.md") if (root / "SKILL.md").exists() else "",
            "script_names": [script.name for script in scripts],
            "script_count": len(scripts),
            "default": bool(metadata.get("default", False)),
            "loaded_from_registry": True,
        }
        return ResearchSkill(
            skill_id=skill_id,
            name=name,
            scenario=scenario,
            trigger_keywords=trigger_keywords,
            required_inputs=required_inputs,
            plan_template=plan_template,
            evaluation_focus=evaluation_focus,
            skill_type=skill_type if skill_type in {"builtin", "pack"} else "pack",
            version=version,
            source_path=_display_path(root),
            instruction_path="SKILL.md" if (root / "SKILL.md").exists() else "",
            instructions_excerpt=instructions_excerpt,
            scripts=scripts,
            metadata=metadata,
        )

    def _resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved


def _looks_like_skill_pack(root: Path) -> bool:
    return (root / "skill.json").exists() or (root / "SKILL.md").exists()


def _heading_from_markdown(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _first_paragraph(markdown: str) -> str:
    blocks = [block.strip() for block in markdown.split("\n\n") if block.strip()]
    for block in blocks:
        if not block.startswith("#"):
            return _trim(block, 220)
    return ""


def _markdown_excerpt(markdown: str, limit: int = 1200) -> str:
    if not markdown:
        return ""
    lines: list[str] = []
    chars = 0
    for line in markdown.splitlines():
        stripped = line.rstrip()
        if not stripped and lines and lines[-1] == "":
            continue
        candidate = stripped if stripped else ""
        if chars + len(candidate) + 1 > limit:
            break
        lines.append(candidate)
        chars += len(candidate) + 1
    return "\n".join(lines).strip()


def _keywords_from_name(name: str) -> list[str]:
    parts = [part for part in re_split_skill_name(name) if part]
    keywords = sorted({name, *parts})
    return keywords


def re_split_skill_name(name: str) -> list[str]:
    import re

    return [part for part in re.split(r"[_\-\s]+", name) if part]


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _minimal_env() -> dict[str, str]:
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "SystemRoot",
        "WINDIR",
        "HOME",
        "USERPROFILE",
        "TEMP",
        "TMP",
        "COMSPEC",
        "PATHEXT",
    }
    env = {name: value for name, value in os.environ.items() if name in allowed}
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _parse_json_output(stdout: str) -> dict[str, Any]:
    payload = stdout.strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"raw_stdout": stdout}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _display_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)
