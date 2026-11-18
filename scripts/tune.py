import argparse, itertools, csv
from pathlib import Path
import yaml
from types import SimpleNamespace
from src.train import run_training

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=["numpy","tensorflow","pytorch"],default="numpy")
    p.add_argument("--model",choices=["linear","mlp"],default="mlp")
    p.add_argument("--config",default="configs/tuning.yaml")
    args=p.parse_args()
    cfg=yaml.safe_load(open(args.config))["training"]
    rows=[]
    for lr,epochs,hidden,activation in itertools.product(
        cfg["learning_rates"],cfg["epochs"],cfg["hidden_sizes"],cfg["activations"]):
        ns=SimpleNamespace(command="train",backend=args.backend,model=args.model,activation=activation,
            epochs=epochs,learning_rate=lr,hidden=hidden,seed=42,data=None,target="y",
            samples=500,noise=.15,test_size=.2)
        metrics=run_training(ns)
        rows.append({"learning_rate":lr,"epochs":epochs,"hidden":hidden,
                     "activation":activation,**metrics})
    rows.sort(key=lambda r:r["rmse"])
    Path("reports").mkdir(exist_ok=True)
    with open("reports/tuning_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("Best configuration:",rows[0])
if __name__=="__main__": main()
