# test_data_loader.py
import unittest
from plasticity import data_loader

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        # Test that data is loaded correctly
        data = data_loader.load_data('path/to/data')
        self.assertIsNotNone(data)

    def test_invalid_path(self):
        # Test that an error is raised when an invalid path is provided
        with self.assertRaises(FileNotFoundError):
            data_loader.load_data('invalid/path')