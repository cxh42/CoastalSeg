import sys

from fetch_assets import fetch_models


def main(argv: list[str]) -> int:
    force = "--force" in argv
    fetch_models(force=force)
    print("Models downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
