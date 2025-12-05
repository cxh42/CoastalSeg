import sys

from fetch_assets import fetch_datasets


def main(argv: list[str]) -> int:
    force = "--force" in argv
    fetch_datasets(force=force)
    print("Datasets downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
