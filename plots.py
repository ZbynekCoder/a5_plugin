import argparse
import json
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_path", type=str, default="outputs/log.jsonl")
    p.add_argument("--out_path", type=str, default="outputs/plot.png")
    p.add_argument("--metric", type=str, default="token_acc")
    return p.parse_args()


def main():
    args = parse_args()
    data = {}

    with open(args.log_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            L = row["len"]
            data.setdefault(L, {"step": [], args.metric: []})
            data[L]["step"].append(row["step"])
            data[L][args.metric].append(row[args.metric])

    for L, d in sorted(data.items()):
        plt.plot(d["step"], d[args.metric], label=f"L={L}")

    plt.xlabel("step")
    plt.ylabel(args.metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_path)
    print(f"saved plot to {args.out_path}")


if __name__ == "__main__":
    main()
