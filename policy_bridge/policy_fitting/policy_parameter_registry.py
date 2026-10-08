from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any
import torch

def save_checkpoint(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _unique_staging_file(target)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

def save_checkpoint_immutable(path: str | Path, value: dict[str, Any]) -> None:
    """Atomically publish a new checkpoint without ever replacing a target.

    Serialization happens in the configured workspace temporary directory.
    A same-filesystem hard link then publishes the completed inode with the
    kernel's create-if-absent semantics.  If another process wins the target
    name race, ``os.link`` raises ``FileExistsError`` and its file is left
    untouched.  The staging file is removed on every exit path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _unique_staging_file(target)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

def save_json_immutable(path: str | Path, value: Any) -> None:
    """Atomically publish a JSON document with create-if-absent semantics."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _unique_staging_file(target)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

def _unique_staging_file(target: Path) -> tuple[int, Path]:
    temp_root = Path(os.environ.get('TMPDIR', '/dat/workspace/hsun/tmp')).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{target.name}.partial-', dir=temp_root)
    return (descriptor, Path(name))

def _publish_new(temporary: Path, target: Path) -> None:
    try:
        os.link(temporary, target, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(f'immutable output already exists: {target}') from error
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

def load_checkpoint(path: str | Path, map_location: str | torch.device='cpu') -> dict[str, Any]:
    value = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not (not isinstance(value, dict) or 'model_state' not in value):
        pass
    else:
        raise ValueError(f'{path} is not a PTCG policy checkpoint')
    return value
