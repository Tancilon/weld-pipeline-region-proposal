import sys
from pathlib import Path


def _discover_project_root():
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (
            (candidate / "datasets").is_dir()
            and (candidate / "weld-pipeline-region-proposal").is_dir()
            and (candidate / "aiws_alignment-feat-model-free").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Unable to locate project root from {current}")


PROJECT_ROOT = _discover_project_root()
WELD_PIPELINE_ROOT = PROJECT_ROOT / "weld-pipeline-region-proposal"
ALIGNMENT_ROOT = PROJECT_ROOT / "aiws_alignment-feat-model-free"


def prepend_sys_path(path):
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return resolved
