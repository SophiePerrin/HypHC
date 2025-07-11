"""Configuration parameters."""

config_args = {
    # training
    "seed": 1234,
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "eval_every": 10,
    "patience": 20,
    "optimizer": "RAdam",
    "save": 1,
    "fast_decoding": 1,
    "num_samples": -1,

    # model
    "dtype": "double",
    "rank": 2,
    "temperature": 0.01,
    "init_size": 1e-3,
    "anneal_every": 20,
    "anneal_factor": 1.0,
    "max_scale": 1 - 1e-3,

    # dataset
    "dataset": "zoo",

    "optimize_alpha": False,                # ##
    "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],  # ##
    "alpha": 0.5,                           # ##

    'no_progress': (False, "Désactive les barres de progression pendant l'entraînement") # ##

}
