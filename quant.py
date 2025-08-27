from hmmlearn.hmm import GaussianHMM
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import seaborn as sns
from arch import arch_model
import cvxpy as cp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split,Dataset
from sklearn.metrics import classification_report, confusion_matrix



tickers = ["SPY","TLT","GLD","BTC-USD","^DJI"]
start_date = "2014-01-01"
end_date = "2024-01-01"

def fetch_log_returns(tickers,start_date,end_date):
    data = yf.download(tickers,start=start_date,end=end_date)["Close"]
    data = data.dropna()
    log_returns = np.log(data/data.shift(1))
    log_returns = log_returns.dropna()
    return log_returns

log_returns  = fetch_log_returns (tickers,start_date,end_date)
print(log_returns)

#log_returns.cumsum().plot(figsize=(12,6), title="Cumulative Log Returns")
#plt.grid(True)
#plt.show()


def train_hmm_model(log_returns, n):
    model = GaussianHMM(n_components=n,covariance_type="diag",n_iter=7700,min_covar=1e-3,algorithm="viterbi",random_state=999)
    model.fit(log_returns.values)
    hidden_states = model.predict(log_returns.values)
    return model,hidden_states

n=5 # number of regimes or states
hmm_model , regime_names = train_hmm_model(log_returns,n)
log_returns["Regimes"] = regime_names
print(log_returns["Regimes"])

for i in range(n):
    mask = log_returns['Regimes'] == i
    vol = log_returns[mask].iloc[:, :-1].std().mean()
    print(f"Regime {i}: Avg Volatility = {vol:.5f}")




def ewma_correlation_matrix(log_returns,lambda_a):
    T,N=log_returns.shape
    cov_matrix = np.zeros((N,N))
    columns  = log_returns.columns
    rt0 = log_returns.iloc[0].values.reshape(-1, 1)
    cov_matrix = rt0 @ rt0.T

    for t in range(1, T):
        rt = log_returns.iloc[t].values.reshape(-1, 1)
        cov_matrix = lambda_a * cov_matrix + (1 - lambda_a) * (rt @ rt.T)
    
    stddevs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(stddevs, stddevs)
    return pd.DataFrame(corr_matrix, index=columns, columns=columns)

lambda_a = 0.94
regime_corr_matrices = {}
regime_cov_matrix = {}
forecasted_vol={}
def forecast_volatility_garch(regime_returns):
    for assets in regime_returns.columns:
        returns = regime_returns[assets].dropna() * 100 # log returns percenages to values 
        model = arch_model(returns, vol ="GARCH",p=1,q=1,o=0,dist="gaussian") #GARCH(1,1)
        res = model.fit(update_freq=1,disp="off")
        forecast =  res.forecast(horizon=1)
        vol = np.sqrt(forecast.variance.iloc[-1,0])/100 #convert back
        forecasted_vol[assets] = vol
    return forecasted_vol
regime_garch_cov_matrices = {}

for k in range(n):
    regime_returns = log_returns[log_returns['Regimes'] == k].iloc[:, :-1]
    ewma_corr = ewma_correlation_matrix(regime_returns, lambda_a)
    regime_corr_matrices[k] = ewma_corr
    ewma_corr_per_reg = regime_corr_matrices[k]
    
    forecasted_vols = forecast_volatility_garch(regime_returns)
    assets = list(forecasted_vols.keys())
    
    D = np.diag([forecasted_vols[asset] for asset in assets])
    cov_mat = D @ ewma_corr_per_reg.loc[assets, assets].values @ D.T
    
    cov_df_n = pd.DataFrame(cov_mat, index=assets, columns=assets)
    regime_garch_cov_matrices[k] = cov_df_n


for k in range(n):
    print(f"\nRegime {k} — GARCH Covariance Matrix:")
    print(regime_garch_cov_matrices[k].round(6))  # show 6 decimal places

expected_returns_per_regime = {}

for k in range(n):
    regime_returns = log_returns[log_returns["Regimes"] == k].iloc[:, :-1]
    mu_k = regime_returns.mean().values  # Expected return vector
    expected_returns_per_regime[k] = mu_k

print (expected_returns_per_regime)
target_annual_return = 0.25   # 25% target return
target_daily_return = target_annual_return / 252

optimized_weights = {}

for k in range(len(expected_returns_per_regime)):
    mu = expected_returns_per_regime[k]
    Sigma = regime_garch_cov_matrices[k].values
    w = cp.Variable(len(mu))
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        mu @ w >= target_daily_return
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimized_weights[k] = w.value
    print(f" Regime {k} ")
    print(pd.Series(w.value, index=regime_garch_cov_matrices[k].columns).round(4))

alpha = 0.97
target_annual_return = 0.33
target_daily_return = target_annual_return / 252
assets = ["BTC-USD", "GLD", "SPY", "TLT", "^DJI"]
n_assets = len(assets)

cvar_weights = {}
cvar_values = {}

for k in range(5):  # Assuming 5 regimes
    regime_data = log_returns[log_returns["Regimes"] == k].iloc[:, :-1].dropna().values
    T = regime_data.shape[0]
    
    w = cp.Variable(n_assets)
    nu = cp.Variable()
    xi = cp.Variable(T)

    portfolio_returns = regime_data @ w

    objective = cp.Minimize(nu + (1 / ((1 - alpha) * T)) * cp.sum(xi))

    constraints = [
        xi >= -portfolio_returns - nu,
        xi >= 0,
        cp.sum(w) == 1,
        w >= 0,
        regime_data.mean(axis=0) @ w >= target_daily_return
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    cvar_weights[k] = w.value
    cvar_values[k] = problem.value

cvar_df = pd.DataFrame(cvar_weights, index=assets)
cvar_df = (cvar_df * 100).round(2)
for regime in cvar_df.columns:
    print(f"\n CVaR Weights — Regime {regime}")
    print(cvar_df[regime])
   
def lagged_ret(log_returns,regimes,time_lag):
    X , y = [],[]
    for t in range(time_lag, len(log_returns)):
        X.append(log_returns.iloc[t - time_lag : t, :-1].values)
        y.append(regimes[t])
    return np.array(X),np.array(y)

regimes = log_returns["Regimes"].values
X,y = lagged_ret(log_returns,regimes,time_lag=10)
print(X.shape,y.shape)
    
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CustomDataset(X, y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
