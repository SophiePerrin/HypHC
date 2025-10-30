"""Training utils."""

import argparse
import hashlib
import os


def str2bool(v):
    """Converts string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_flags_from_config(parser, config_dict):
    """Adds a flag (and default value) to an ArgumentParser for each parameter in a config."""

    def OrNone(default):
        def func(x):
            if x.lower() == "none":
                return None
            elif default is None:
                return str(x)
            else:
                return type(default)(x)
        return func

    for param in config_dict:
        # ✅ Extraire (default, help) si c’est un tuple, sinon valeur seule
        raw_value = config_dict[param]
        if isinstance(raw_value, tuple):
            default, help_text = raw_value
        else:
            default, help_text = raw_value, f"Default: {raw_value}"

        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)

            elif isinstance(default, bool):
                parser.add_argument(
                    f"--{param}",
                    action="store_true",
                    help=help_text
                )

            elif isinstance(default, list) and default:
                parser.add_argument(
                    f"--{param}",
                    type=type(default[0]),
                    nargs='+',
                    default=default,
                    help=help_text
                )

            else:
                parser.add_argument(
                    f"--{param}",
                    type=OrNone(default),
                    default=default,
                    help=help_text
                )

        except argparse.ArgumentError:
            print(f"Could not add flag for param {param} because it was already present.")

    return parser


'''
def add_flags_from_config(parser, config_dict):
    """Adds a flag (and default value) to an ArgumentParser for each parameter in a config."""

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
                
            elif isinstance(default, bool):
                # ✅ NOUVEAU : ajout explicite des booléens comme flags (--flag active le bool)
                parser.add_argument(
                    f"--{param}",
                    action="store_true",
                    help=f"Default: {default}"
                )

            elif isinstance(default, list) and default:
                # ✅ NOUVEAU : ajout des listes avec nargs='+'
                parser.add_argument(
                    f"--{param}",
                    type=type(default[0]),  # ex : float pour [0.0, 0.5, 1.0]
                    nargs='+',
                    default=default
                )

            else:
                parser.add_argument(f"--{param}", type=OrNone(default), default=default)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser
'''


def hash_dict(values):
    """Hash of dict key, value pairs."""
    m = hashlib.sha256()
    keys = sorted(list(values.keys()))
    for k in keys:
        if k != "seed":
            m.update(str(values[k]).encode('utf-8'))
    return m.hexdigest()


def get_savedir(args):
    """Hash of args used for training."""
    dir_hash = hash_dict(args.__dict__)
    save_dir = os.path.join(os.environ["SAVEPATH"], args.dataset, dir_hash)
    return save_dir
