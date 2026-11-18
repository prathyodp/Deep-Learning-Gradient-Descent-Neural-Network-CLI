import argparse
from .train import run_training

def parser():
    p=argparse.ArgumentParser(prog="dlcli", description="Gradient descent and neural network experiment CLI")
    sub=p.add_subparsers(dest="command", required=True)
    t=sub.add_parser("train", help="train a regression model")
    t.add_argument("--backend", choices=["numpy","tensorflow","pytorch"], default="numpy")
    t.add_argument("--model", choices=["linear","mlp"], default="linear")
    t.add_argument("--activation", choices=["relu","sigmoid","tanh","linear"], default="tanh")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--learning-rate", type=float, default=0.03)
    t.add_argument("--hidden", type=int, default=16)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--data", default=None)
    t.add_argument("--target", default="y")
    t.add_argument("--samples", type=int, default=500)
    t.add_argument("--noise", type=float, default=0.15)
    t.add_argument("--test-size", type=float, default=0.2)
    t.set_defaults(func=run_training)
    e=sub.add_parser("evaluate", help="show metrics from the most recent run")
    e.add_argument("--model-path", default="artifacts/run.json")
    def evaluate(args):
        import json
        with open(args.model_path) as f: print(json.dumps(json.load(f),indent=2))
    e.set_defaults(func=evaluate)
    return p

def main():
    args=parser().parse_args()
    if args.command=="train": args.func(args)
    else: args.func(args)

if __name__=="__main__": main()
