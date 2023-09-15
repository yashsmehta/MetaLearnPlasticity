import plasticity.trainer as trainer
from omegaconf import OmegaConf
import numpy as np


if __name__ == "__main__":
    coeff_mask = np.zeros((3, 3, 3))
    coeff_mask[1, 1, 0] = 1
    coeff_mask[1, 0, 0] = 1
    coeff_mask[0, 1, 0] = 1
    coeff_mask[0, 0, 0] = 1

    cfg_dict = {
        "seed": 1,
        "num_exps": 10,
        "num_blocks": 3,
        "reward_ratios": ((0.2, 0.8), (0.8, 0.2), (0.2, 0.8)),
        "trials_per_block": 80,
        "input_dim": 2,
        "output_dim": 1,
        "num_epochs": 1000,
        "moving_avg_window": 10,
        "coeff_mask": coeff_mask.tolist(),
    }

    cfg = OmegaConf.create(cfg_dict)
    assert (
        len(cfg.reward_ratios) == cfg.num_blocks
    ), "length of reward ratios should be equal to number of blocks!"
    print("passing config to trainer...")

    trainer.train(cfg)
