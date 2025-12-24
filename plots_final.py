
import argparse
import json
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_path", type=str, default="outputs/log_final.jsonl")
    p.add_argument("--out_path", type=str, default="outputs/plot_final.png")
    p.add_argument("--model", type=str, default="", help="optional: filter by model name (plugin/gru/select/adapter)")
    return p.parse_args()


def main():
    args = parse_args()
    data = {}  # key: (model, L)

    with open(args.log_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = row.get("model", "unknown")
            if args.model and m != args.model:
                continue
            L = row["len"]
            key = (m, L)
            data.setdefault(key, {"step": [], "final_acc": []})
            data[key]["step"].append(row["step"])
            data[key]["final_acc"].append(row["final_acc"])

    for (m, L), d in sorted(data.items(), key=lambda x: (x[0][0], x[0][1])):
        plt.plot(d["step"], d["final_acc"], label=f"{m}-L={L}")

    plt.xlabel("step")
    plt.ylabel("final_acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_path)
    print(f"saved plot to {args.out_path}")


if __name__ == "__main__":
    main()
