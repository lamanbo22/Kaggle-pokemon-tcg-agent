"""Prepare a reviewable runtime release without modifying the source archive.

Only transparent exception wrappers and generated layout markers are removed.
Python's AST unparser normalizes formatting; all branches and expressions remain.
The resulting AST is checked against the transformed source AST before writing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


MARKER = "__equivalent_layout_profile__"
ASSETS = (
    "candidate_policy.pt",
    "opponent_routed_policy.pt",
    "deck.csv",
    "search_card_registry.json",
    "opponent_strategy_priors.json",
)
NATIVE_ENGINE = "engine_bridge/battle_runtime_bridge.so"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_inventory(source: Path) -> dict[str, Path]:
    result = {}
    for path in source.rglob("*.py"):
        module = path.relative_to(source).with_suffix("").as_posix().replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-9]
        result[module] = path
    return result


def import_names(module: str, path: Path, tree: ast.AST) -> set[str]:
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
                stem_parts = package[:len(package) - node.level + 1]
                if node.module:
                    stem_parts.append(node.module)
                stem = ".".join(stem_parts)
            else:
                stem = node.module or ""
            result.add(stem)
            result.update(stem + "." + alias.name for alias in node.names)
    return result


def runtime_closure(source: Path) -> tuple[dict[str, Path], dict[str, ast.AST]]:
    modules = module_inventory(source)
    trees = {name: ast.parse(path.read_text(encoding="utf-8-sig")) for name, path in modules.items()}
    selected = set()
    pending = ["main", "engine_bridge.engine_battle_state"]
    while pending:
        module = pending.pop()
        if module in selected:
            continue
        selected.add(module)
        for imported in import_names(module, modules[module], trees[module]):
            if imported in modules:
                pending.append(imported)
            parts = imported.split(".")
            for index in range(1, len(parts)):
                ancestor = ".".join(parts[:index])
                if ancestor in modules:
                    pending.append(ancestor)
    return {name: modules[name] for name in sorted(selected)}, trees


class RemoveGeneratedWrappers(ast.NodeTransformer):
    def __init__(self) -> None:
        self.wrapper_count = 0
        self.marker_count = 0

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == MARKER:
            # Generated markers are a three-element literal tuple, never executable expressions.
            value = ast.literal_eval(node.value)
            if not (isinstance(value, tuple) and len(value) == 3 and value[:2] == ("bridge", 17)):
                raise ValueError("Unexpected layout marker value")
            self.marker_count += 1
            return None
        return self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        self.generic_visit(node)
        transparent_handler = not node.handlers
        if len(node.handlers) == 1:
            handler = node.handlers[0]
            transparent_handler = (
                handler.type is None
                and handler.name is None
                and len(handler.body) == 1
                and isinstance(handler.body[0], ast.Raise)
                and handler.body[0].exc is None
                and handler.body[0].cause is None
            )
        if not node.orelse and transparent_handler and all(isinstance(item, ast.Pass) for item in node.finalbody):
            self.wrapper_count += 1
            return node.body
        return node


def make_release(source: Path, destination: Path) -> dict:
    source = source.absolute()
    destination = destination.absolute()
    if not (source / "submission_manifest.json").is_file():
        raise ValueError("Source submission manifest is missing")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Release and source must be separate directories")
    modules, trees = runtime_closure(source)
    for name in modules:
        if any(isinstance(node, ast.Name) and node.id == MARKER and isinstance(node.ctx, ast.Load) for node in ast.walk(trees[name])):
            raise ValueError("A runtime file reads generated layout metadata")
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, path in modules.items():
        relative = path.relative_to(source)
        transformer = RemoveGeneratedWrappers()
        expected = ast.fix_missing_locations(transformer.visit(ast.parse(path.read_text(encoding="utf-8-sig"))))
        rendered = ast.unparse(expected) + "\n"
        actual = ast.parse(rendered)
        if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
            raise ValueError(f"AST changed during serialization: {relative}")
        compile(actual, str(relative), "exec")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8", newline="\n")
        entries.append({
            "path": relative.as_posix(),
            "kind": "normalized_python",
            "original_sha256": digest(path),
            "release_sha256": digest(target),
            "removed_transparent_wrappers": transformer.wrapper_count,
            "removed_layout_markers": transformer.marker_count,
            "normalized_ast_verified": True,
        })
    for relative in (*ASSETS, NATIVE_ENGINE):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        original_hash = digest(source / relative)
        if digest(target) != original_hash:
            raise ValueError(f"Asset copy differs: {relative}")
        entries.append({
            "path": relative,
            "kind": "ignored_local_validation_engine" if relative == NATIVE_ENGINE else "unchanged_asset",
            "original_sha256": original_hash,
            "release_sha256": original_hash,
        })
    (destination / "requirements.txt").write_text(
        "# CPU inference also requires the compatible native engine; see README.md.\n"
        "numpy>=1.26\n"
        "torch>=2.6\n",
        encoding="utf-8", newline="\n",
    )
    (destination / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n.venv/\n.pytest_cache/\n"
        "# Native engine supplied separately; redistribution status is unverified.\n"
        "engine_bridge/*.so\nengine_bridge/*.dll\nengine_bridge/*.dylib\n"
        "# Local evaluation runs.\nartifacts/\n",
        encoding="utf-8", newline="\n",
    )
    scripts = destination / "tools"
    scripts.mkdir(exist_ok=True)
    if Path(__file__).resolve() != (scripts / "prepare_release.py").resolve():
        shutil.copyfile(__file__, scripts / "prepare_release.py")
    provenance = {
        "schema_version": 1,
        "scope": "runtime_and_inference_assets",
        "source_archive": "User-provided Pokemon submission archive (validated extraction)",
        "source_submission_manifest_sha256": digest(source / "submission_manifest.json"),
        "selection": "Transitive static imports from main.py plus engine_battle_state.py, including deferred imports and package initializers.",
        "source_python_files": len(module_inventory(source)),
        "release_python_runtime_files": len(modules),
        "transform_scope": [
            "Remove try wrappers with no else, no handlers or one bare except that only reraises, and no finally or a finally containing only pass.",
            "Remove assignments to the generated __equivalent_layout_profile__ tuple after confirming no selected module reads that name.",
            "Normalize Python formatting with ast.unparse; preserve conditions, imports, executable expressions, and all other statements.",
        ],
        "validation": "Each normalized file compiles and its AST matches the deliberately transformed original AST. Asset SHA-256 values match. Runtime equivalence tests are recorded separately.",
        "semantic_limits": "Source line numbers, formatting, exception tracing events, and generated layout metadata change. No claim of identical debugging or introspection behavior.",
        "excluded": [
            "Unreferenced training, diagnostic, packaging, and experiment scripts",
            "Whitespace-only original readme",
            "Original inventories containing local machine paths",
            "policy_entry_bootstrap.py (undefined init placeholder, not imported by main.py)",
        ],
        "native_engine": {
            "path": NATIVE_ENGINE,
            "git_ignored": True,
            "purpose": "Local validation only",
            "redistribution_license": "unverified",
            "required_exports": ["GameInitialize", "BattleStart", "SearchBegin", "SearchStep", "SearchEnd", "SearchRelease"],
        },
        "files": entries,
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    result = make_release(args.source, args.destination)
    print(json.dumps({
        "runtime_python_files": result["release_python_runtime_files"],
        "removed_wrappers": sum(row.get("removed_transparent_wrappers", 0) for row in result["files"]),
        "removed_markers": sum(row.get("removed_layout_markers", 0) for row in result["files"]),
        "destination": str(args.destination.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
