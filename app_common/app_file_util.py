from pathlib import Path


class AppFileUtil:
    """Utility class for file operations in the app_common module."""

    import os
    from pathlib import Path

    @staticmethod
    def resolve_path(path_str: str, __file__val) -> str:
        """Resolve relative paths safely from repo root."""
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        # repo root = 3 levels up (training/tasks/ -> training/ -> project root)
        root = Path(__file__val).resolve().parent.parent.parent
        return str((root / p).resolve())
