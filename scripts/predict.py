"""Run a PEP/NRC entrypoint on authorized JSONL inputs, entirely offline."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['pep', 'nrc'], required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen:
            raise ValueError('Every input must have a unique nonempty string id')
        seen.add(row['id'])
        for key in ['model_id', 'problem']:
            if not isinstance(row.get(key), str):
                raise ValueError(f'Missing or invalid {key}')
        if not isinstance(row.get('reasoning_effort', 'default'), str):
            raise ValueError('Invalid reasoning_effort')
    spec = importlib.util.spec_from_file_location(
        f'aimo_{args.method}', ROOT / 'methods' / args.method / 'solution.py')
    method = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = method
    spec.loader.exec_module(method)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row['model_id'], row.get('reasoning_effort', 'default')), []).append(index)
    output = [None] * len(rows)
    for (model, effort), indices in groups.items():
        predictions = method.are_robust(model, effort, [rows[i]['problem'] for i in indices])
        if len(predictions) != len(indices) or any(type(p) is not bool for p in predictions):
            raise ValueError('Method violated the official prediction contract')
        for index, prediction in zip(indices, predictions):
            output[index] = dict(id=rows[index]['id'], is_robust=prediction, valid=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row) + '\n' for row in output))
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
