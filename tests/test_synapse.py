import unittest
from plasticity import synapse

class TestSynapse(unittest.TestCase):
    def test_synapse_initialization(self):
        # Test that the synapse is initialized correctly
        my_synapse = synapse.MySynapse()
        self.assertIsNotNone(my_synapse)

    def test_synapse_activation(self):
        # Test that the synapse activates correctly
        my_synapse = synapse.MySynapse()
        activation = my_synapse.activate('input')
        self.assertIsNotNone(activation)