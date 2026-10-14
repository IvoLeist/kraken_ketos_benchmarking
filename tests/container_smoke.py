"""Exercise compilation, worker-based training and plotting on tiny synthetic pages.

This checks deployment plumbing, not benchmark performance or OCR accuracy.
"""
import json
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    output = Path('artifacts') / f'container-check-{uuid.uuid4().hex[:10]}'
    source = output / 'input'
    source.mkdir(parents=True)
    namespace = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
    ET.register_namespace('', namespace)
    for index in range(2):
        image = Image.new('RGB', (240, 160), 'white')
        draw = ImageDraw.Draw(image)
        root = ET.Element(f'{{{namespace}}}PcGts')
        page = ET.SubElement(root, f'{{{namespace}}}Page', imageFilename=f'{index}.jpg',
                             imageWidth='240', imageHeight='160')
        region = ET.SubElement(page, f'{{{namespace}}}TextRegion', id='r1')
        ET.SubElement(region, f'{{{namespace}}}Coords', points='10,10 220,10 220,150 10,150')
        for line_index, top in enumerate((20, 90)):
            draw.text((20, top), 'hello', fill='black', font_size=28)
            line = ET.SubElement(region, f'{{{namespace}}}TextLine', id=f'l{line_index}')
            ET.SubElement(line, f'{{{namespace}}}Coords',
                          points=f'15,{top-5} 140,{top-5} 140,{top+40} 15,{top+40}')
            ET.SubElement(line, f'{{{namespace}}}Baseline', points=f'15,{top+28} 140,{top+28}')
            text = ET.SubElement(line, f'{{{namespace}}}TextEquiv')
            ET.SubElement(text, f'{{{namespace}}}Unicode').text = 'hello'
        image.save(source / f'{index}.jpg')
        ET.ElementTree(root).write(source / f'{index}.xml', encoding='utf-8', xml_declaration=True)
    datasets, runs, plots = (output / name for name in ('datasets', 'runs', 'plots'))
    subprocess.run([sys.executable, '-m', 'scripts.datasets', '--source', str(source),
                    '--output', str(datasets), '--sizes', '1', '--validation-pages', '1'], check=True)
    subprocess.run([sys.executable, '-m', 'scripts.benchmark', '--datasets', str(datasets),
                    '--output', str(runs), '--sizes', '1', '--workers', '1', '--threads', '1',
                    '--batch-sizes', '2', '--epochs', '3', '--max-width', '512',
                    '--torch-compile', 'disabled', '--optimizer', 'AdamW'], check=True)
    summary_path, = runs.glob('*/summary.json')
    summary = json.loads(summary_path.read_text())
    if summary['returncode'] != 0 or summary['interrupted']:
        raise RuntimeError('Container training check did not complete')
    log = (summary_path.parent / 'process.log').read_text()
    if 'max_epochs=3' not in log or not list(summary_path.parent.glob('checkpoints/*.safetensors')):
        raise RuntimeError('Three-epoch completion or exported weights missing')
    subprocess.run([sys.executable, '-m', 'scripts.plot', '--input', str(runs),
                    '--datasets', str(datasets), '--output', str(plots)], check=True)
    for name in ('results.csv', 'compilation.png', 'dataset-vs-wall_seconds.png'):
        if not (plots / name).is_file():
            raise RuntimeError(f'Missing plot output: {name}')
    print(f'Container integration check passed: {output}', flush=True)


if __name__ == '__main__':
    main()
