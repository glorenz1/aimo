"""Package a standalone PEP or NRC Small Models Track submission."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(method, output):
    directory = ROOT / 'methods' / method
    files = sorted(p for p in directory.iterdir() if p.suffix in {'.py', '.json', '.txt'})
    required = {'solution.py', 'classifier.json', 'small.txt'}
    if not required.issubset({p.name for p in files}):
        raise ValueError('Incomplete submission')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            entry = zipfile.ZipInfo(path.name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, path.read_bytes())
    temporary.replace(output)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['pep', 'nrc'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'{package(args.method, args.output)}  {args.output}')


if __name__ == '__main__':
    main()
