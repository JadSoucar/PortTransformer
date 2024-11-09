from PT import *
import json


def training_loop(train_loader,SEQ_LENGTH,BATCH_SIZE,EMBED_DIM,NUM_ASSETS,num_heads,num_epochs,lr,weight_decay,model_name,save=True):
    
    model = PortfolioTransformer(input_dim=NUM_ASSETS, embed_dim=EMBED_DIM, num_heads=num_heads, num_layers=4, num_assets=NUM_ASSETS)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize the previous weights for the first batch as a uniform portfolio 
    prev_weights = 1/NUM_ASSETS * torch.ones(BATCH_SIZE, NUM_ASSETS)  

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            weights = model(batch_x)

            # Calculate loss
            loss = sharpe_loss(weights, batch_y, prev_weights)
            loss.backward()

            # Apply gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Update prev_weights
            prev_weights = weights.detach()

        #Store Avg Loss
        avg_loss = epoch_loss / len(train_loader)

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")


    if save:
        torch.save(model.state_dict(), model_name)

    return avg_loss



if __name__ == '__main__':

    '''
    Train in accordance with the expanding window training protocol outlined in the paper 
    ALSO use the optimized hyperpramers
    '''

    with open('Models/params.json','r') as f:
        trial_params = json.load(f)

    #For 7 Years of Data
    SEQ_LENGTH = trial_params['seq_length']  # Sequence length (number of past days to consider) (i.e Tau)
    BATCH_SIZE = trial_params['batch_size']   # Batch size for training
    EMBED_DIM = trial_params['embed_dim']   # Embedding dimension in Time2Vec and model
    NUM_ASSETS = 9  # Number of assets in the portfolio
    num_epochs = 100
    num_heads = trial_params['num_heads'] 
    lr = trial_params['lr'] 
    weight_decay = trial_params['weight_decay'] 

    #Load Data
    returns = pd.read_csv('returns.csv').to_numpy()

    #Initial Training Cycle from 2007 - 2015
    returns = returns.T[1:].T
    train_data = returns[:252*8]  
    train_dataset = FinancialDataset(train_data, seq_length=SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    training_loop(train_loader,SEQ_LENGTH,BATCH_SIZE,EMBED_DIM,NUM_ASSETS,num_heads,num_epochs,lr,weight_decay,'Models/pt_2015.pth') #Model named for the year its trained on

    #Training with an expanding window after 2015
    for ix in range(9,17):
        year = 2007 + ix

        print('#'*10, year, '#'*10)
        train_data = returns[:252*(ix)]  
        train_dataset = FinancialDataset(train_data, seq_length=SEQ_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        training_loop(train_loader,SEQ_LENGTH,BATCH_SIZE,EMBED_DIM,NUM_ASSETS,num_heads,num_epochs,lr,weight_decay,f'Models/pt_{year}.pth')
    