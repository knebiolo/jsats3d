"""Basic tests for the jsats3d package."""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestJsats3dBasic(unittest.TestCase):
    def test_import_jsats3d(self):
        import jsats3d
        self.assertTrue(hasattr(jsats3d, 'create_project_db'))
        self.assertTrue(hasattr(jsats3d, 'temp_interpolator'))

if __name__ == '__main__':
    unittest.main()
