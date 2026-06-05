#!/usr/bin/env python3
import argparse
import ast
import configparser
import glob
import math
import os
import sys
import tempfile
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="breakout")
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--total-agents", type=int, default=128)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--cpu-inference", action="store_true")
    parser.add_argument("--train-fp16", action="store_true")
    parser.add_argument("--skip-save-load", action="store_true")
    return parser.parse_args()


def flatten(prefix, value):
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}/{key}" if prefix else key
            yield from flatten(next_prefix, nested)
    else:
        yield prefix, value


def finite_metrics(metrics):
    bad = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            bad.append((key, value))
    return bad


def parse_value(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(env_name):
    default_path = os.path.join(ROOT, "config", "default.ini")
    config_paths = glob.glob(os.path.join(ROOT, "config", "**", "*.ini"), recursive=True)

    if env_name == "default":
        parser = configparser.ConfigParser()
        parser.read(default_path)
    else:
        parser = None
        for path in config_paths:
            candidate = configparser.ConfigParser()
            candidate.read([default_path, path])
            if env_name in candidate["base"]["env_name"].split():
                parser = candidate
                break
        if parser is None:
            raise ValueError(f"No config for env_name {env_name}")

    args = defaultdict(dict)
    for section in parser.sections():
        for key, raw_value in parser[section].items():
            value = parse_value(raw_value)
            if section == "base":
                args[key] = value
            else:
                args[section][key] = value

    args["env_name"] = env_name
    for section in parser.sections():
        args.setdefault(section, {})
    return dict(args)


def validate_config(args):
    minibatch_size = args["train"]["minibatch_size"]
    horizon = args["train"]["horizon"]
    total_agents = args["vec"]["total_agents"]
    if minibatch_size % horizon != 0:
        raise ValueError(f"minibatch_size {minibatch_size} must be divisible by horizon {horizon}")
    if minibatch_size > horizon * total_agents:
        raise ValueError(
            f"minibatch_size {minibatch_size} > total_agents {total_agents} * horizon {horizon}"
        )


def configure(base_args, cli, temp_root):
    args = dict(base_args)
    args["checkpoint_dir"] = os.path.join(temp_root, "checkpoints")
    args["log_dir"] = os.path.join(temp_root, "logs")
    args["checkpoint_interval"] = max(cli.steps * 10, 1)
    args["wandb"] = False
    args["profile"] = False

    args["vec"] = dict(args["vec"])
    args["vec"]["total_agents"] = cli.total_agents
    args["vec"]["num_buffers"] = cli.num_buffers
    args["vec"]["num_threads"] = cli.num_threads

    args["train"] = dict(args["train"])
    args["train"]["total_timesteps"] = cli.steps
    args["train"]["horizon"] = cli.horizon
    args["train"]["minibatch_size"] = cli.minibatch_size
    args["train"]["overlap"] = int(cli.overlap)
    args["train"]["cpu_inference"] = int(cli.cpu_inference)
    args["train"]["train_fp16"] = int(cli.train_fp16)
    args["train"]["gpus"] = 1

    return args


def run_smoke():
    cli = parse_args()
    base_args = load_config(cli.env)

    from pufferlib import _C

    compiled_env = getattr(_C, "env_name", None)
    if compiled_env != cli.env:
        raise RuntimeError(f"compiled _C env_name={compiled_env!r}, expected {cli.env!r}")

    with tempfile.TemporaryDirectory(prefix="pufferlib-metal-smoke-") as temp_root:
        args = configure(base_args, cli, temp_root)
        validate_config(args)

        print(
            "metal_smoke:",
            f"env={cli.env}",
            f"precision_bytes={getattr(_C, 'precision_bytes', None)}",
            f"gpu={getattr(_C, 'gpu', None)}",
            f"agents={args['vec']['total_agents']}",
            f"buffers={args['vec']['num_buffers']}",
            f"threads={args['vec']['num_threads']}",
            f"horizon={args['train']['horizon']}",
            f"minibatch={args['train']['minibatch_size']}",
            f"overlap={args['train']['overlap']}",
            f"cpu_inference={args['train']['cpu_inference']}",
            f"train_fp16={args['train']['train_fp16']}",
        )

        state = _C.create_pufferl(args)
        try:
            model_size = state.num_params()
            print(f"metal_smoke: params={model_size}")

            for idx in range(cli.iterations):
                _C.rollouts(state)
                _C.train(state)
                logs = dict(flatten("", dict(_C.log(state))))
                bad_metrics = finite_metrics(logs)
                if bad_metrics:
                    raise RuntimeError(f"non-finite metrics after iteration {idx}: {bad_metrics}")
                print(
                    "metal_smoke:",
                    f"iteration={idx + 1}",
                    f"steps={logs.get('agent_steps')}",
                    f"sps={logs.get('SPS')}",
                    f"loss_total={logs.get('loss/total')}",
                )

            if not cli.skip_save_load:
                weight_path = os.path.join(temp_root, "weights.bin")
                _C.save_weights(state, weight_path)
                if not os.path.getsize(weight_path):
                    raise RuntimeError("save_weights wrote an empty file")
                _C.load_weights(state, weight_path)
                print(f"metal_smoke: save_load_bytes={os.path.getsize(weight_path)}")
        finally:
            _C.close(state)

    print("metal_smoke: ok")


if __name__ == "__main__":
    run_smoke()
