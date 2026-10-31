import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/download_soup_model.py'
SPEC = importlib.util.spec_from_file_location('model_downloader_tested', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloaderIntegrityTests(unittest.TestCase):
    def test_same_size_non_lfs_corruption_is_rejected(self):
        content = b'author'
        entry = {'size': len(content), 'blobId': hashlib.sha1(b'blob 6\0' + content).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_bytes(b'attakr')
            self.assertIsNone(MODULE.verified_digest(path, entry))
            path.write_bytes(content)
            self.assertEqual(MODULE.verified_digest(path, entry), hashlib.sha256(content).hexdigest())

    def test_completed_partial_recovers_without_network(self):
        content = b'weights'
        entry = {'rfilename': 'model.safetensors', 'size': len(content),
                 'lfs': {'sha256': hashlib.sha256(content).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.safetensors.part').write_bytes(content)
            with patch.object(MODULE, 'DEST', root), patch.object(MODULE.urllib.request, 'urlopen') as network:
                MODULE.download(entry, 'a' * 40)
                network.assert_not_called()
            self.assertEqual((root / 'model.safetensors').read_bytes(), content)

    def test_bad_partial_is_preserved_for_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.part'
            path.write_bytes(b'bad')
            MODULE.quarantine_partial(path)
            self.assertFalse(path.exists())
            files = list(Path(directory).glob('weights.part.invalid-*'))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), b'bad')


if __name__ == '__main__':
    unittest.main()
