"""Select nested PAGE/JPG subsets and compile each with measured ketos calls."""
import argparse
import hashlib
import json
import random
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa

from scripts.monitor import measure


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def discover(source):
    pairs = []
    images = set()
    for xml in sorted(Path(source).resolve().rglob('*.xml')):
        root = ET.parse(xml).getroot()
        page = next((x for x in root.iter() if x.tag.rsplit('}', 1)[-1] == 'Page'), None)
        if page is None:
            raise ValueError(f'Not PAGE XML: {xml}')
        image = (xml.parent / page.attrib['imageFilename']).resolve()
        if image.suffix.lower() not in ('.jpg', '.jpeg') or not image.is_file():
            raise ValueError(f'Missing JPG referenced by {xml}: {image}')
        if image in images:
            raise ValueError(f'Multiple annotations refer to the same image: {image}')
        images.add(image)
        if not any(x.text and x.text.strip() for x in root.iter() if x.tag.rsplit('}', 1)[-1] == 'Unicode'):
            raise ValueError(f'No transcription in {xml}')
        pairs.append((xml, image))
    return pairs


def select(pairs, sizes, validation_pages, seed):
    if not sizes or min(sizes) < 1 or validation_pages < 1:
        raise ValueError('Page counts must be positive')
    required = max(sizes) + validation_pages
    if len(pairs) < required:
        raise ValueError(f'Need {required} distinct PAGE/JPG pairs; found {len(pairs)}')
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    validation = pairs[:validation_pages]
    training = pairs[validation_pages:]
    return {str(n): training[:n] for n in sorted(set(sizes))}, validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('training_data'))
    p.add_argument('--output', type=Path, default=Path('artifacts/datasets'))
    p.add_argument('--sizes', nargs='+', type=int, default=[5, 10, 15])
    p.add_argument('--validation-pages', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    subsets, validation = select(discover(args.source), args.sizes, args.validation_pages, args.seed)
    subsets['validation'] = validation
    args.output.mkdir(parents=True, exist_ok=True)
    # Invalidate old metadata until every dataset has successfully compiled.
    (args.output / 'datasets.json').unlink(missing_ok=True)
    records = {}
    ketos = str(Path(sys.executable).parent / 'ketos')
    for name, pairs in subsets.items():
        target = (args.output / f'{name}.arrow').resolve()
        temporary = target.with_suffix('.partial.arrow')
        temporary.unlink(missing_ok=True)
        manifest = args.output / f'{name}.xml.txt'
        manifest.write_text(''.join(f'{xml}\n' for xml, _ in pairs))
        command = [ketos, '--workers', str(args.workers), '--threads', str(args.threads),
                   'compile', '--format-type', 'page', '--output', str(temporary),
                   *[str(xml) for xml, _ in pairs]]
        run_dir = args.output / 'compile-runs' / f'{name}-{uuid.uuid4().hex[:10]}'
        result = measure(command, run_dir, dict(kind='compile', pages=len(pairs), dataset=name,
                         workers=args.workers, threads=args.threads, seed=args.seed))
        if result['returncode'] or not temporary.is_file():
            raise RuntimeError(f'Compilation failed; see {run_dir / "process.log"}')
        with pa.memory_map(str(temporary), 'r') as stream:
            reader = pa.ipc.open_file(stream)
            lines = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        if not lines:
            raise RuntimeError(f'No text lines extracted; see {run_dir / "process.log"}')
        temporary.replace(target)
        records[name] = dict(path=str(target), pages=len(pairs), lines=lines, bytes=target.stat().st_size,
                             sha256=sha256(target), compile_run=str(run_dir.resolve()),
                             sources=[dict(xml=str(x), image=str(i), xml_sha256=sha256(x),
                                           image_sha256=sha256(i)) for x, i in pairs])
        (args.output / f'{name}.txt').write_text(str(target) + '\n')
        print(f'{name}: {len(pairs)} pages, {target.stat().st_size / 2**20:.2f} MiB', flush=True)
    (args.output / 'datasets.json').write_text(json.dumps(dict(seed=args.seed, datasets=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
