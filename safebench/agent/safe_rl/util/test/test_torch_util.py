import unittest
import numpy as np
import torch
from torch._C import device

from safe_rl.util.torch_util import combined_shape, discount_cumsum, export_device_env_variable, get_torch_device, to_ndarray, to_tensor, to_device


class TestTorchUtils(unittest.TestCase):
    """ Testing the torch util.
    """
    def test_combined_shape(self):
        input_sample = {"length": 3, "shape": (4, 5)}
        expected_output = (3, 4, 5)
        actual_output = combined_shape(**input_sample)

        self.assertTrue(np.allclose(expected_output, actual_output, rtol=1e-2))

    def test_discount_cumsum(self):
        input_sample = {"x": np.array([1, 1, 1]), "discount": 0.9}
        expected_output = np.array([1 + 0.9 + 0.81, 1 + 0.9, 1])
        actual_output = discount_cumsum(**input_sample)

        self.assertTrue(np.allclose(expected_output, actual_output, rtol=1e-2))

    def test_device(self):
        if torch.cuda.is_available():
            export_device_env_variable("gpu")
            device = get_torch_device()
            self.assertTrue(device == torch.device("cuda:0"))
            a = to_tensor([1., 2., 3.], device=device)
            self.assertTrue(a.device == torch.device("cuda:0"))
            self.assertTrue(
                np.allclose(to_ndarray(a), np.array([1., 2., 3.]), rtol=1e-2))
            self.assertTrue(to_device(a, "cpu").device == torch.device("cpu"))

        export_device_env_variable("cpu", 1)
        device = get_torch_device()
        self.assertTrue(device == torch.device("cpu"))
        a = to_tensor([1., 2., 3.], device=device)
        self.assertTrue(a.device == torch.device("cpu"))
        self.assertTrue(
            np.allclose(to_ndarray(a), np.array([1., 2., 3.]), rtol=1e-2))

        # ******************* test to_device ********************

        self.assertTrue(to_device(a, "cpu").device == torch.device("cpu"))
        export_device_env_variable("gpu")
        a = to_device(a, get_torch_device())
        self.assertTrue(a.device == get_torch_device())


if __name__ == '__main__':
    unittest.main()
