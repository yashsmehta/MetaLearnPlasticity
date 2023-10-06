import plasticity.trainer as trainer
from omegaconf import OmegaConf
import numpy as np


if __name__ == "__main__":
    coeff_mask = np.zeros((3, 3, 3))
    coeff_mask[:, :, 0] = 1
    coeff_mask[:, :, 1] = 1

    cfg_dict = {
        "num_train": 3,
        "num_eval": 5,
        "num_epochs": 250,
        "trials_per_block": 80,
        "log_interval": 25,  # log training data every x epochs
        "num_blocks": 3,
        "reward_ratios": ((0.2, 0.8), (0.9, 0.1), (0.2, 0.8)),
        "log_expdata": True,
        "use_experimental_data": False,
        "flyid": 1,
        "fit_data": "behavior",  # searches for words: "neural", "behavior"
        "neural_recording_sparsity": 1.0,  # sparsity of 1. means all neurons are recorded
        "measurement_noise_scale": 0.0,  # scale of gaussian noise added to neural recordings would be measurement_noise * firing_rate
        "layer_sizes": "[2, 10, 1]",  # [input_dim, hidden_dim, output_dim], only 2, 3 layer network supported
        "input_firing_mean": 0.75,
        "input_variance": 0.05,
        "l1_regularization": 1e-2,  # 5e-2 # only for volterra model
        "generation_coeff_init": "X1R1W0",  # "X1R1W0"
        "generation_model": "volterra",  # "volterra" or "mlp
        "plasticity_coeff_init": "random",  # "random", "zeros" or "reward"
        "plasticity_model": "volterra",  # "volterra" or "mlp
        "meta_mlp_layer_sizes": [3, 10, 1],  # [3, hidden_dim, 1]
        "moving_avg_window": 10,
        "data_dir": "../data/",
        "log_dir": "logs/",
        "trainable_coeffs": int(np.sum(coeff_mask)),
        "coeff_mask": coeff_mask.tolist(),
        "exp_name": "eval",
        "reward_term": "reward",
    }

    cfg = OmegaConf.create(cfg_dict)

    config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, config)

    trainer.train(cfg)
    # evaluate.simulate_model(cfg)
