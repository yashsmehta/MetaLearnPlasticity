import unittest
import os
from plasticity import utils

class TestUtils(unittest.TestCase):
    def test_save_logs(self):
        # Test that logs are saved correctly
        utils.save_logs('cfg', 'df')
        self.assertTrue(os.path.exists('path/to/log'))

    def test_experiment_list_to_tensor(self):
        # Test that the experiment list is converted to a tensor correctly
        tensor = utils.experiment_list_to_tensor('longest_trial_length', 'nested_list', 'list_type')
        self.assertIsNotNone(tensor)