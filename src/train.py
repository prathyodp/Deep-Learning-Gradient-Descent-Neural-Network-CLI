from pathlib import Path
import json, csv, time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .data import load_data
from .models import NumpyNetwork, train_autograd

def run_training(args):
    Xtr, Xte, ytr, yte, df = load_data(
        path=args.data, target=args.target, samples=args.samples, noise=args.noise,
        seed=args.seed, test_size=args.test_size)
    start=time.time()
    if args.backend == "numpy":
        net=NumpyNetwork(Xtr.shape[1], args.model, args.hidden, args.activation, args.seed)
        history=net.fit(Xtr, ytr, args.epochs, args.learning_rate)
        predict=net.predict
        state=net.state()
    else:
        predict, history=train_autograd(args.backend, Xtr, ytr, args.model, args.hidden,
                                        args.activation, args.epochs, args.learning_rate, args.seed)
        state={"backend": args.backend, "model": args.model, "activation": args.activation}
    pred=predict(Xte)
    metrics={"mse": float(mean_squared_error(yte, pred)),
             "mae": float(mean_absolute_error(yte, pred)),
             "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
             "r2": float(r2_score(yte, pred))}
    Path("artifacts").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    state.update({"backend": args.backend, "model": args.model, "activation": args.activation,
                  "feature_count": int(Xtr.shape[1]), "target": args.target})
    with open("artifacts/model.json","w") as f: json.dump(state,f,indent=2)
    np.savetxt("artifacts/test_predictions.csv", np.column_stack([yte.ravel(), pred.ravel()]),
               delimiter=",", header="actual,predicted", comments="")
    run={"backend":args.backend,"model":args.model,"activation":args.activation,
         "epochs":args.epochs,"learning_rate":args.learning_rate,"hidden":args.hidden,
         "seed":args.seed,"metrics":metrics,"final_train_loss":history[-1],
         "duration_seconds":round(time.time()-start,3)}
    with open("artifacts/run.json","w") as f: json.dump(run,f,indent=2)
    log_path=Path("reports/experiment_log.csv")
    exists=log_path.exists()
    with log_path.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["backend","model","activation","epochs","learning_rate",
                                      "hidden","final_train_loss","mse","mae","rmse","r2","duration_seconds"])
        if not exists: w.writeheader()
        w.writerow({**{k:run[k] for k in ["backend","model","activation","epochs","learning_rate","hidden","final_train_loss","duration_seconds"]}, **metrics})
    print(json.dumps({"metrics":metrics,"final_train_loss":history[-1],
                      "duration_seconds":run["duration_seconds"]},indent=2))
    return metrics
