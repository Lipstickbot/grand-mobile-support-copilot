"""Download only author Qwen3-8B weights/configs at a recorded commit, with hashes.

No model code is executed. Partial downloads resume using HTTP byte ranges.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / '.private/soup-setup/qwen3-8b-metadata.json'
DEST = ROOT / '.private/models/qwen3-8b'
ALLOWED = {'LICENSE', 'config.json', 'generation_config.json', 'merges.txt',
           'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verified_digest(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return None
    checksum = digest(path)
    lfs_sha = entry.get('lfs', {}).get('sha256')
    if lfs_sha:
        return checksum if checksum == lfs_sha else None
    blob = entry.get('blobId')
    if not blob or len(blob) != 40:
        raise ValueError('Author metadata lacks an expected file hash')
    git_hash = hashlib.sha1(f'blob {entry["size"]}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            git_hash.update(chunk)
    return checksum if git_hash.hexdigest() == blob else None


def quarantine_partial(path):
    quarantined = path.with_name(path.name + f'.invalid-{time.time_ns()}')
    path.replace(quarantined)
    print(f'QUARANTINED {path.name}', flush=True)


def download(entry, revision):
    name = entry['rfilename']
    if '/' in name or '\\' in name or name.startswith('.'):
        raise ValueError(f'Unsafe model filename: {name}')
    target = DEST / name
    partial = DEST / (name + '.part')
    size = entry['size']
    checksum = verified_digest(target, entry)
    if checksum:
        print(f'CHECKED {name}', flush=True)
        return {'filename': name, 'bytes': size, 'sha256': checksum}
    url = f'https://huggingface.co/Qwen/Qwen3-8B/resolve/{revision}/{name}?download=true'
    for attempt in range(4):
        try:
            if partial.exists() and partial.stat().st_size >= size:
                checksum = verified_digest(partial, entry)
                if checksum:
                    partial.replace(target)
                    return {'filename': name, 'bytes': size, 'sha256': checksum}
                quarantine_partial(partial)
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'GrandMobileLocalExperiment/0.1'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    offset = 0
                elif offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected Content-Range; refusing corrupt resume')
                mode = 'ab' if offset else 'wb'
                done = offset
                last_log = time.monotonic()
                with partial.open(mode) as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last_log >= 30:
                            print(f'DOWNLOAD {name}: {done / 1e9:.2f}/{size / 1e9:.2f} GB', flush=True)
                            last_log = time.monotonic()
            if partial.stat().st_size != size:
                raise RuntimeError(f'Unexpected size for {name}: {partial.stat().st_size}, expected {size}')
            checksum = verified_digest(partial, entry)
            if not checksum:
                quarantine_partial(partial)
                raise RuntimeError(f'Author hash mismatch for {name}; quarantined and restarting')
            partial.replace(target)
            print(f'COMPLETE {name}', flush=True)
            return {'filename': name, 'bytes': size, 'sha256': checksum}
        except Exception as exc:
            print(f'RETRY {name} ({attempt + 1}/4): {type(exc).__name__}: {exc}', flush=True)
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    metadata = json.loads(META.read_text(encoding='utf-8'))
    if metadata.get('id') != 'Qwen/Qwen3-8B':
        raise ValueError('Unexpected model identity')
    revision = metadata['sha']
    if len(revision) != 40 or not all(c in '0123456789abcdef' for c in revision):
        raise ValueError('Expected exact model commit')
    DEST.mkdir(parents=True, exist_ok=True)
    entries = [entry for entry in metadata['siblings']
               if entry['rfilename'] in ALLOWED or entry['rfilename'].endswith('.safetensors')]
    print(f'Author model revision {revision}; {sum(e["size"] for e in entries) / 1e9:.2f} GB', flush=True)
    completed = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download, entry, revision) for entry in entries]
        for future in as_completed(futures):
            completed.append(future.result())
    manifest = {'model_id': metadata['id'], 'revision': revision,
                'files': sorted(completed, key=lambda e: e['filename']),
                'trust_remote_code': False}
    (DEST / 'download-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('MODEL DOWNLOAD VERIFIED', flush=True)


if __name__ == '__main__':
    main()
