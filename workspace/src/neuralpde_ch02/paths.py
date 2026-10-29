"""Keep revision outputs separate from the frozen Chapter 2 package."""
from pathlib import Path


def output_directory(path):
    """Resolve an output path and reject writes into the frozen chapter."""
    output = Path(path).resolve()
    frozen_chapter = Path(__file__).resolve().parents[2] / "chapters/ch02"
    frozen_chapter = frozen_chapter.resolve()
    if output == frozen_chapter or frozen_chapter in output.parents:
        raise ValueError("The original ch02 package is frozen; choose a new output directory")
    return output
