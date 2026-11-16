"""Average TCN probabilities from several checkpoints into one logits file."""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode", choices=("arithmetic", "geometric"),
        default="arithmetic")
    args = parser.parse_args()

    probabilities = []
    for path in args.inputs:
        with np.load(path) as bundle:
            probabilities.append(bundle["probabilities"].astype(np.float64))
    probabilities = np.clip(np.stack(probabilities, axis=0), 1e-8, 1.0)
    if args.mode == "geometric":
        logp = np.mean(np.log(probabilities), axis=0)
        mean = np.exp(logp - logp.max(axis=1, keepdims=True))
    else:
        mean = np.mean(probabilities, axis=0)
    mean = np.clip(mean, 1e-8, 1.0)
    mean /= mean.sum(axis=1, keepdims=True)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        logits=np.log(mean).astype(np.float32),
        probabilities=mean.astype(np.float32),
    )
    print("Wrote {} from {} models".format(args.out, len(probabilities)))


if __name__ == "__main__":
    main()
