# Define the objective function for Optuna
from train import training_loop
from PT import *
import optuna
import json


def objective(trial):
    # Hyperparameters to tune
    embed_dim = trial.suggest_int("embed_dim", 32, 128, step=32)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16, 32])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    seq_length = trial.suggest_int("seq_length", 7, 30, step=7) #1 weeks to a moonth


    #Load Data
    returns = pd.read_csv('returns.csv').to_numpy()
    returns = returns.T[1:].T
    train_data = returns[:int(len(returns)*.75)]  
    train_dataset = FinancialDataset(train_data, seq_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = returns[int(len(returns)*.75):]  

    avg_loss = training_loop(train_loader,seq_length,batch_size,embed_dim,9,num_heads,50,lr,weight_decay,None,False)

    return np.abs(avg_loss + 2) #Setting a Sharpe Ratio of 2 as a target


if __name__ == '__main__':

    # Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    # Display the best parameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Loss: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


    with open('Models/params.json','w') as f:
        json.dump(trial.params,f)



#ADJUST EPOCH AND STUDY VALUES FOR FINAL RUN
#TUNE
#LINE 26 --> 50
#LIN# 35 --> 100
#TRAIN
#LINE 68 --> 100