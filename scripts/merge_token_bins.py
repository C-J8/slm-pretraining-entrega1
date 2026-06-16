import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import ensure_dir


def copy_file(src: Path, dst_handle, chunk_size: int):
    total = src.stat().st_size
    with open(src, "rb") as src_handle:
        with tqdm(total=total, unit="B", unit_scale=True, desc=src.name) as pbar:
            while True:
                chunk = src_handle.read(chunk_size)
                if not chunk:
                    break
                dst_handle.write(chunk)
                pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--chunk-mb", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output)
    inputs = [Path(path) for path in args.inputs]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    ensure_dir(output.parent)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_output.unlink(missing_ok=True)
    chunk_size = args.chunk_mb * 1024 * 1024

    with open(tmp_output, "wb") as dst_handle:
        for src in inputs:
            copy_file(src, dst_handle, chunk_size)

    tmp_output.replace(output)
    total_tokens = output.stat().st_size // 2
    print(f"wrote {output} | tokens={total_tokens:,}")


if __name__ == "__main__":
    main()
