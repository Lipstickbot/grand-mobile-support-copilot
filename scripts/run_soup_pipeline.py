"""Wait for the author download, verify it, then execute the bounded local experiment.

This foreground process does not schedule future work. It stops on the first failure.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from download_soup_model import META, DEST, ALLOWED, verified_digest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / '.private/soup-setup'
MANIFEST = DEST / 'download-manifest.json'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.time()
    state = {'status': 'waiting_for_download', 'stages': []}
    def save():
        (SETUP / 'pipeline.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    save()
    while not MANIFEST.is_file():
        if time.time() - started > 3600:
            raise RuntimeError('Author download did not complete within one hour')
        time.sleep(5)
    state['status'] = 'verifying_author_hashes'
    save()
    metadata = json.loads(META.read_text(encoding='utf-8'))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['revision'] != metadata['sha']:
        raise RuntimeError('Model revision mismatch')
    for entry in metadata['siblings']:
        name = entry['rfilename']
        if name not in ALLOWED and not name.endswith('.safetensors'):
            continue
        checksum = verified_digest(DEST / name, entry)
        if checksum is None:
            raise RuntimeError(f'Author hash verification failed: {name}')
        print(f'AUTHOR HASH VERIFIED {name}', flush=True)
    for stage in ('preflight', 'train', 'reload'):
        record = {'stage': stage, 'status': 'running', 'started_at_unix': time.time()}
        state['stages'].append(record)
        state['status'] = stage
        save()
        log_path = SETUP / f'{stage}-run.log'
        print(f'START {stage}; log: {log_path}', flush=True)
        env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u',
                                        str(ROOT / 'scripts/soup_experiment.py'), stage],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            save()
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError(f'{stage} exceeded the 30-minute experiment limit')
        record.update(status='passed' if code == 0 else 'failed', exit_code=code,
                      seconds=round(time.time() - record['started_at_unix'], 3))
        save()
        print(f'END {stage}: exit {code}', flush=True)
        if code:
            state['status'] = 'failed'
            save()
            return code
    state.update(status='complete', total_seconds=round(time.time() - started, 3))
    save()
    print('LOCAL EXPERIMENT COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        state_path = SETUP / 'pipeline.json'
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            state.update(status='failed', error=str(exc))
            state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f'PIPELINE FAILED: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
