import matplotlib.pyplot as plt
from PT import *
import json

def evaluate_portfolio_value(model, num_assets, batch_size, seq_length, returns_data, initial_value=1000):
    model.eval()  # Set the model to evaluation mode
    portfolio_values = [initial_value]

    # Prepare the data for evaluation
    dataset = FinancialDataset(returns_data, seq_length=seq_length)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size 1 for sequential evaluation

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # Forward pass to get portfolio weights for the current timestep
            weights = model(batch_x).squeeze(0)  # Shape: [seq_length, num_assets]
            weights = torch.sign(weights) * torch.softmax(torch.abs(weights), dim=-1)  # Normalize weights

            # Calculate the portfolio return based on the current weights and returns
            current_returns = batch_y[:, -1, :]  # Shape: [1, num_assets] (last returns in the sequence)
            portfolio_return = torch.sum(weights * current_returns)

            # Update the portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value.item())

            # Update prev_weights for the next iteration
            prev_weights = weights.detach()

    return portfolio_values


if __name__ == '__main__':
    #Load in parameters and data
    with open('Models/params.json','r') as f:
        trial_params = json.load(f)

    embed_dim = trial_params['embed_dim']   # Embedding dimension in Time2Vec and model
    num_assets = 9  # Number of assets in the portfolio
    num_heads = trial_params['num_heads'] 
    seq_length = trial_params['seq_length']
    batch_size = trial_params['batch_size']
    returns = pd.read_csv('returns.csv').to_numpy()

    #Validate from 2016 - 2024, Using the previous year's model for each year
    all_portfolio_values = []
    initial_value = 1000

    for ix in range(8,17):
        year = 2007 + ix #Year Last Trained, 
        model = PortfolioTransformer(input_dim=num_assets, embed_dim=embed_dim, num_heads=num_heads, num_layers=4, num_assets=num_assets)
        model.load_state_dict(torch.load(f'Models/pt_{year}.pth'))

        # Example usage with validation data
        val_data = returns[252*ix : 252*(ix+1)].astype(np.float32)  # Predict Return of year + 1
        val_data = val_data.T[1:].T
        portfolio_values = evaluate_portfolio_value(model, num_assets, batch_size, seq_length, val_data, initial_value=initial_value)
        all_portfolio_values.append(portfolio_values)
        initial_value = portfolio_values[-1]

    # Plot the portfolio value over time to visualize performance
    all_portfolio_values_vec = np.hstack(all_portfolio_values)
    plt.plot(all_portfolio_values_vec )
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value ($)")
    plt.show()

    plt.savefig("port_value.png")  # Save as PNG
