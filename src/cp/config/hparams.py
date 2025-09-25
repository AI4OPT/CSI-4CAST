default_hparams = {
    "process": "csi_pred",
    "mode": {
        "train": False,
        "test": True,
        "vis": True,
    },
    "model": {
        "name": "PM4",
        "dir": "z_weights/tdd/PM4",
    },
    "dataset": {
        "is_U2D": False,
        "path_dict_noise_info": "z_outputs/noise/dict_res.json",
        "noise_type": "vanilla",  # ['phase', 'gaussian', 'quant', 'packagedrop', 'burst']
    },
    "train_and_valid_dataset": {
        "dir_dataset": "z_shared/Training_Dataset",
        "batch_size": 512,
        "is_few": False,
        "train_ratio": 0.9,
        "valid_ratio": 0.1,
    },
    "test_dataset": {
        "dir_dataset": "z_shared/Testing_Dataset",
        "batch_size": 512,
        "is_zero": False,
    },
    "output": "z_outputs",
    "optimizer": {
        "name": "Adam",
        "params": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "eps": 1e-08,
            "betas": (0.9, 0.999),
        },
    },
    # "scheduler": {
    #     "name": "StepLR",
    #     "params": {
    #         "step_size": 150,
    #         "gamma": 0.1,
    #     }
    # },
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "params": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 0.0001,
            "cooldown": 0,
            "min_lr": 1e-06,
        },
    },
    "early_stopping": {
        "patience": 50,
        "min_delta": 0.0001,
    },
    "loss": {
        "name": "NMSE",
        "params": {},
    },
    "training": {
        "num_epochs": 5,
        "valid_period": 1,
        "model_save_period": 10,
    },
    "model_params": {
        "LLM4CP": {
            "name": "LLM4CP",
            "params": {
                "gpt_type": "gpt2",
                "d_ff": 768,
                "d_model": 768,
                "gpt_layers": 6,
                "pred_len": 4,
                "prev_len": 16,
                "use_gpu": 1,
                "gpu_id": 0,
                "mlp": 0,
                "res_layers": 4,
                "K": 48,
                "UQh": 1,
                "UQv": 1,
                "BQh": 1,
                "BQv": 1,
                "patch_size": 4,
                "stride": 1,
                "res_dim": 64,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
            },
        },
        "Transformer": {
            "name": "Transformer",
            "params": {
                "enc_in": 96,
                "dec_in": 96,
                "c_out": 96,
                "prev_len": 16,
                "label_len": 12,
                "out_len": 4,
                "factor": 5,
                "d_model": 92,
                "n_heads": 8,
                "e_layers": 3,
                "d_layers": 2,
                "d_ff": 192,
                "dropout": 0.05,
                "attn": "full",
                "embed": "fixed",
                "activation": "gelu",
                "SR_rate": 6,
                "interpolate_f": "linear",
                "output_attention": False,
                "distil": True,
            },
        },
        "CNN": {
            "name": "CNN",
            "params": {},
        },
        "RNN": {
            "name": "RNN",
            "params": {
                "features": 96,
                "input_size": 96,
                "hidden_size": 192,
                "num_layers": 4,
                "pred_len": 4,
            },
        },
        "LSTM": {
            "name": "LSTM",
            "params": {
                "features": 96,
                "input_size": 96,
                "hidden_size": 192,
                "num_layers": 4,
                "pred_len": 4,
            },
        },
        "GRU": {
            "name": "GRU",
            "params": {
                "features": 96,
                "input_size": 96,
                "hidden_size": 192,
                "num_layers": 4,
                "pred_len": 4,
            },
        },
        "NP": {
            "name": "NP",
            "params": {
                "pred_len": 4,
            },
        },
        "PAD": {
            "name": "PAD",
            "params": {
                "p": 8,
                "startidx": 16,
                "subcarriernum": 48,
                "Nt": 16,
                "Nr": 1,
                "pred_len": 4,
            },
        },
        "PM1": {
            "name": "PM1",
            "params": {
                "patch_size": 4,
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "pred_len": 4,
                "mlp_hidden_dim": 1024,
                "mlp_num_layers": 2,
                "mlp_dropout": 0.1,
            },
        },
        "PM2": {
            "name": "PM2",
            "params": {
                "patch_size": 4,
                "hist_len": 16,
                "pred_len": 4,
                "dim_data": 96,
                "mlp_hidden_dim": 1024,
                "mlp_num_layers": 5,
                "mlp_dropout": 0.1,
            },
        },
        "PM3": {
            "name": "PM3",
            "params": {
                "patch_size": 4,
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "pred_len": 4,
                "attn_num_layers": 2,
                "attn_num_heads": 4,
                "attn_dropout": 0.1,
            },
        },
        "PM4": {
            "name": "PM4",
            "params": {
                "patch_size": 4,
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
            },
        },
        "PM5": {
            "name": "PM5",
            "params": {
                "patch_size": 4,
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "pred_len": 4,
                "mlp_hidden_dim": 1024,
                "mlp_num_layers": 2,
                "mlp_dropout": 0.1,
            },
        },
        "PM6": {
            "name": "PM6",
            "params": {
                "patch_size": 4,
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
                "wavelet": "db1",
                "mode": "symmetric",
                "threshold_scale": 1,
            },
        },
        "PM7": {
            "name": "PM7",
            "params": {
                # preprocessor
                # MIP instead of the patch
                # embedding
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                # predictor
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
            },
        },
        "PM8": {
            "name": "PM8",
            "params": {
                # preprocessor
                # MIP Mixer
                "temporal_proj_num_layers": 2,
                "temporal_proj_hidden_dim": 256,
                "arl_num_layers": 2,
                "arl_hidden_dim": 256,
                # embedding
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                # predictor
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
            },
        },
        "PM9": {
            "name": "PM9",
            "params": {
                # preprocessor
                "patch_size": 4,
                # embedding
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "res_ratio": 4,  # GhostNet
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                # predictor
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
            },
        },
        "PM10": {
            "name": "PM10",
            "params": {
                # preprocessor
                "patch_size": 4,
                # embedding
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "res_groups": 4,  # ShuffleNet
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                # predictor
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
            },
        },
        "PM11": {
            "name": "PM11",
            "params": {
                # preprocessor
                "patch_size": 4,
                # embedding
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                # predictor
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
                # denoiser
                "list_n_filters_2d": [2, 8, 16, 32, 64],
                "list_filter_sizes_2d": [3, 3, 3, 3, 3, 3, 3, 3],
                "filter_sizes_1d": 3,
                "activation": "tanh",  # relu, tanh, sigmoid, gelu
            },
        },
        "PM12": {
            "name": "PM12",
            "params": {
                # preprocessor
                "patch_size": 4,
                # embedding
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                # predictor
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
                # mask parameters
                "train_missing_rate": 0.2,
                "min_missing_rate": 0.01,
                "max_missing_rate": 0.1,
                "learnable_mask": False,
            },
        },
        "PM13": {
            "name": "PM13",
            "params": {
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
            },
        },
        "PM14": {
            "name": "PM14",
            "params": {
                # masked handler
                "mh_dim_model": 512,
                "mh_num_layers": 2,
                "mh_num_heads": 4,
                "mh_hidden_dim": 1024,
                "mh_dropout": 0.1,
                "mh_train_missing_ratio": 0.2,
                "mh_min_missing_ratio": 0.01,
                "mh_max_missing_ratio": 0.2,
                "mh_learnable_mask": False,
                "mh_TFI_hidden_dim": 512,
                "mh_TFI_num_layers": 2,
                # preprocessor
                "patch_size": 4,
                # embedding
                "dim_model": 768,
                "num_res_layers": 4,
                "res_dim": 64,
                "hist_len": 16,
                "dim_data": 96,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                # predictor
                "pred_len": 4,
                "trans_num_layers": 2,
                "trans_num_heads": 4,
                "trans_hidden_dim": 1024,
                "trans_dropout": 0.1,
            },
        },
    },
}

if __name__ == "__main__":
    import argparse
    import copy
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="LLM4CP")
    parser.add_argument("--model_dir", "-md", type=str, default="")
    parser.add_argument("--fdd", "-f", action="store_true", help="FDD")
    args = parser.parse_args()

    hparams = copy.deepcopy(default_hparams)
    hparams["model"]["name"] = args.model
    hparams["model"]["dir"] = args.model_dir
    hparams["dataset"]["is_U2D"] = args.fdd
    U2D = "FDD" if args.fdd else "TDD"

    path_hparams_json = "CP/hparams/{}_md_{}_{}.json".format(args.model, args.model_dir.replace("/", "-"), U2D)

    with open(path_hparams_json, "w") as f:
        json.dump(hparams, f, indent=4)

    print("Hparams saved to {}".format(path_hparams_json))
