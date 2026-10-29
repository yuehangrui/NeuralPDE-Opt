import json
from pathlib import Path
import tempfile
import unittest
from neuralpde_book.workspace import initialize,verify,payload


class WorkspaceTests(unittest.TestCase):
    def test_fresh_workspace_is_complete_without_parent_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=initialize(Path(tmp)/'standalone')
            self.assertGreater(len(verify(root)['files']),200)
            self.assertFalse((root/'references/legacy_source').exists())
            self.assertTrue((root/'src/neuralpde_ch02/models.py').is_file())
            self.assertTrue((root/'third_party/ch03_upstream/deepxde/LICENSE').is_file())

    def test_existing_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): initialize(tmp)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=initialize(Path(tmp)/'run')
            (root/'book_driver.py').write_text('changed')
            with self.assertRaises(ValueError):verify(root)

    def test_manifest_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=initialize(Path(tmp)/'run')
            manifest=json.loads((root/'PAYLOAD_MANIFEST.json').read_text(encoding='utf-8'))
            manifest['files']['../outside']='0'*64
            (root/'PAYLOAD_MANIFEST.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify(root)

    def test_new_holdout_streams_and_no_old_weight_reuse(self):
        root=payload()
        self.assertIn('2026102803',(root/'chapters/ch03/heldout.py').read_text(encoding='utf-8'))
        self.assertIn('[20261028, 444, 20261029, 1]',(root/'chapters/ch04_delivery/protocol.py').read_text(encoding='utf-8'))
        self.assertIn('[20261028, 5051, 20261029]',(root/'chapters/ch05_delivery/protocol.py').read_text(encoding='utf-8'))
        self.assertIn('20261207',(root/'chapters/ch04_depth/assessment.py').read_text(encoding='utf-8'))
        self.assertNotIn('--reuse-seed0',(root/'book_driver.py').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
