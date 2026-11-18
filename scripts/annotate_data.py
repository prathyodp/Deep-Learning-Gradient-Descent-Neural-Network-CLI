import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description="Annotate numeric regression targets into quantile bands.")
    p.add_argument("--input", required=True)
    p.add_argument("--target", default="y")
    p.add_argument("--output", default="data/processed/annotated.csv")
    args=p.parse_args()
    df=pd.read_csv(args.input)
    if args.target not in df: raise SystemExit(f"Missing target column: {args.target}")
    # Quantile labels are descriptive annotations; original numeric target is retained.
    df["target_band"]=pd.qcut(df[args.target], q=3, labels=["low","medium","high"], duplicates="drop")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output,index=False)
    print(f"Annotated {len(df)} rows -> {args.output}")
if __name__=="__main__": main()
