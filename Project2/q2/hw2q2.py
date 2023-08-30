# Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV into a DataFrame
df = pd.read_csv('forbes2000_data.csv')

# Convert each column to an array
sotck_market_price = []
times = []
col_num = 0
for columns in df.columns:
    if col_num == 0:
        times = columns
        col_num += 1
        continue
    
    current_company = df[columns].to_numpy()
    sotck_market_price.append(current_company)
    col_num += 1

# remove nan's
sotck_market_price = np.array(sotck_market_price)
sotck_market_price[np.isnan(sotck_market_price)] = 0.0

# ============= calculate daily return ================
def compute_daily_return(market_prices):
    daily_returns = []
    
    for i in range(len(market_prices) - 1):
        if market_prices[i] == 0:
            # can't buy at this day, no daily return
            dr = 0.0
        else:
            dr = market_prices[i + 1] / market_prices[i] - 1
        
        daily_returns.append(dr)
    
    return daily_returns

company_daily_returns = []

for i in range(len(sotck_market_price)):
    company_daily_returns.append(compute_daily_return(sotck_market_price[i]))
    
# print(company_daily_returns[1][0:2])

# ========= need to calculate rho ===============
U, s, V_t = np.linalg.svd(company_daily_returns)

# s contains sigma values
# need to calculate rho and plot rho instead ==============
# Create the plot
indices = np.arange(len(s))
plt.plot(indices, s)
# need to calculate rho and plot rho instead =====================

# Add labels and title
plt.xlabel("Index")
plt.ylabel("Rate")
plt.title("Rho")

# Show the plot
plt.show()