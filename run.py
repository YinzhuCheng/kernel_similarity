import os
import sys


def _ensure_package_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.join(repo_root, "kernel_similarity-cursor-api-711f")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def main() -> None:
    _ensure_package_path()
    from kernel_similarity.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
