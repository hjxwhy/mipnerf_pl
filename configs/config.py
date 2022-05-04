import yaml
import argparse
import os
from ast import literal_eval
from os.path import dirname, join, isdir
from typing import Tuple
import shutil
import glob

DEFAULT_CONFIG_FILE = join(dirname(__file__), 'default.yaml')


def _parse_dict(d, d_out=None, prefix=""):
    if d is None:
        return {}
    d_out = d_out if d_out is not None else {}
    for k, v in d.items():
        if isinstance(v, dict):
            _parse_dict(v, d_out, prefix=prefix + k + '.')
        else:
            if isinstance(v, str):
                try:
                    v = literal_eval(v)  # try to parse
                except (ValueError, SyntaxError):
                    pass  # v is really a string

            if isinstance(v, list):
                v = tuple(v)
            d_out[prefix + k] = v
    if prefix == "":
        return d_out


def load(fname):
    with open(fname, 'r') as fp:
        return _parse_dict(yaml.safe_load(fp))


def merge_from_config(config, config_merge):
    for k, v in config_merge.items():
        # assert k in config, f"The key {k} is not in the base config for the merge."
        # if not k in config:
            # print("Add new args {} to config".format(k))
        config[k] = v


def merge_from_file(config, fname):
    merge_from_config(config, load(fname))


def merge_from_list(config, list_merge):
    assert len(list_merge) % 2 == 0, "The list must have key value pairs."
    config_merge = _parse_dict(dict(zip(list_merge[0::2], list_merge[1::2])))
    merge_from_config(config, config_merge)


def default():
    return load(DEFAULT_CONFIG_FILE)


def parse_args(parser: argparse.ArgumentParser) -> Tuple[str, dict, str, argparse.Namespace]:
    args = parser.parse_args()

    # out_dir = os.path.realpath(args.out_dir)
    # if args.rm_out_dir:
    #     if os.path.exists(out_dir):
    #         shutil.rmtree(out_dir)

    # This is ddp save
    # os.makedirs(out_dir, exist_ok=True)
    # regular_files = tuple(sorted(glob.iglob(join(out_dir, '**/*'), recursive=True)))
    # regular_files = tuple(x for x in regular_files if not isdir(x))
    # if len(regular_files) != 0:
    #     msg = f"The output directory {out_dir} is not empty, it has {len(regular_files)} files." \
    #           f" Due to the specifics of pytorch lightning and ddp we only support" \
    #           f" empty/non-existent output directories.\n\n" \
    #           f"The files are: {regular_files}."
    #     # raise RuntimeError(msg)

    config = default()
    config_path = args.config

    if config_path is not None:
        merge_from_file(config, config_path)
    if args.opts is not None:
        merge_from_list(config, args.opts)
    args_dict = args.__dict__
    for k, v in args_dict.items():
        # if not k in config:
            config[k] = v
    return config


# parser = argparse.ArgumentParser()
# parser.add_argument("--config", help="Path to config file.", default='/home/hjx/mipnerf_pl/configs/lego.yaml')
# cfg = parse_args(parser)
# print(cfg)
