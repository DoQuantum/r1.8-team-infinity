"""
Classical Portfolio Optimization Framework with VaR/CVaR Risk Measures
This implementation serves as the foundation for quantum-classical hybrid optimization
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Classical Portfolio Optimizer with VaR and CVaR risk measures
    Designed to be extended with quantum computing capabilities
    """
    
    def __init__(self, tickers, start_date, end_date, confidence_level=0.95):
        """
        Initialize the portfolio optimizer
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date for historical data (YYYY-MM-DD)
        end_date : str
            End date for historical data (YYYY-MM-DD)
        confidence_level : float
            Confidence level for VaR/CVaR calculations (e.g., 0.95 for 95%)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Data containers
        self.prices = None
        self.returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.optimal_weights = None
        
        # Risk metrics
        self.var_historical = None
        self.cvar_historical = None
        
    # def fetch_data(self):
    #     """Fetch historical price data from yfinance"""
    #     print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date}")
        
    #     data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        
    #     if isinstance(data, pd.Series):
    #         data = data.to_frame()
        
    #     self.prices = data
    #     self.returns = data.pct_change().dropna()
        
    #     print(f"Data fetched successfully. Shape: {self.prices.shape}")
    #     return self.prices
    def fetch_data(self):
        """Fetch historical price data (Close prices) from yfinance"""
        print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date}")
        
        # Download data with auto_adjust=True to get adjusted close prices
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        
        # Handle different data structures
        if isinstance(data.columns, pd.MultiIndex):
            # Multiple tickers - select 'Close' prices
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            else:
                # Fallback: if no 'Close', use the first price column
                price_columns = [col for col in data.columns.get_level_values(0) 
                               if col in ['Close', 'Adj Close', 'Open', 'High', 'Low']]
                if price_columns:
                    data = data[price_columns[0]]
                else:
                    raise ValueError("No suitable price columns found in the data")
        else:
            # Single ticker - ensure it's a DataFrame
            if len(self.tickers) == 1:
                data = data[['Close']] if 'Close' in data.columns else data.iloc[:, [0]]
            else:
                # This shouldn't happen, but handle it gracefully
                data = data.iloc[:, :len(self.tickers)]
        
        self.prices = data
        self.returns = data.pct_change().dropna()
        
        print(f"Data fetched successfully. Shape: {self.prices.shape}")
        print(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
        print(f"Number of trading days: {len(self.prices)}")
        
        return self.prices
    def calculate_statistics(self):
        """Calculate expected returns and covariance matrix"""
        self.expected_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252  # Annualized
        
        return self.expected_returns, self.cov_matrix
    
    def calculate_var_historical(self, weights, returns=None):
        """
        Calculate historical VaR for given weights
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        returns : DataFrame, optional
            Returns data (uses self.returns if not provided)
        """
        if returns is None:
            returns = self.returns
        
        portfolio_returns = (returns * weights).sum(axis=1)
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        return -var  # Return positive VaR
    
    def calculate_cvar_historical(self, weights, returns=None):
        """
        Calculate historical CVaR (Expected Shortfall) for given weights
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        returns : DataFrame, optional
            Returns data (uses self.returns if not provided)
        """
        if returns is None:
            returns = self.returns
        
        portfolio_returns = (returns * weights).sum(axis=1)
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        # CVaR is the expected value of returns below VaR
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return -cvar  # Return positive CVaR
    
    def calculate_var_parametric(self, weights, holding_period=1):
        """
        Calculate parametric VaR assuming normal distribution
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        holding_period : int
            Holding period in days
        """
        portfolio_return = np.dot(weights, self.expected_returns) / 252 * holding_period
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(holding_period/252)
        
        z_score = norm.ppf(1 - self.confidence_level)
        var = -(portfolio_return + z_score * portfolio_std)
        
        return var
    
    def calculate_cvar_parametric(self, weights, holding_period=1):
        """
        Calculate parametric CVaR assuming normal distribution
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        holding_period : int
            Holding period in days
        """
        portfolio_return = np.dot(weights, self.expected_returns) / 252 * holding_period
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(holding_period/252)
        
        z_score = norm.ppf(1 - self.confidence_level)
        pdf_z = norm.pdf(z_score)
        
        cvar = -portfolio_return + portfolio_std * (pdf_z / (1 - self.confidence_level))
        
        return cvar
    
    def portfolio_performance(self, weights):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def optimize_min_cvar(self, method='historical', risk_free_rate=0.02):
        """
        Optimize portfolio to minimize CVaR
        
        Parameters:
        -----------
        method : str
            'historical' or 'parametric' CVaR calculation
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
        """
        n_assets = len(self.tickers)
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds (0 <= weight <= 1 for each asset)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Objective function
        if method == 'historical':
            objective = lambda x: self.calculate_cvar_historical(x)
        else:
            objective = lambda x: self.calculate_cvar_parametric(x)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.optimal_weights = result.x
        return result.x
    
    def optimize_mean_cvar(self, lambda_param=0.5, method='historical', 
                          min_return=None, max_cvar=None):
        """
        Optimize portfolio with mean-CVaR objective
        
        Parameters:
        -----------
        lambda_param : float
            Trade-off parameter between return and CVaR (0 to 1)
            0 = minimize CVaR only, 1 = maximize return only
        method : str
            'historical' or 'parametric' CVaR calculation
        min_return : float, optional
            Minimum required return constraint
        max_cvar : float, optional
            Maximum allowed CVaR constraint
        """
        n_assets = len(self.tickers)
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Add optional constraints
        if min_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: np.dot(x, self.expected_returns) - min_return
            })
        
        if max_cvar is not None:
            if method == 'historical':
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: max_cvar - self.calculate_cvar_historical(x)
                })
            else:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: max_cvar - self.calculate_cvar_parametric(x)
                })
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Objective function (minimize negative return + CVaR)
        if method == 'historical':
            objective = lambda x: -lambda_param * np.dot(x, self.expected_returns) + \
                                (1 - lambda_param) * self.calculate_cvar_historical(x)
        else:
            objective = lambda x: -lambda_param * np.dot(x, self.expected_returns) + \
                                (1 - lambda_param) * self.calculate_cvar_parametric(x)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.optimal_weights = result.x
        return result.x
    
    def optimize_sharpe_with_cvar_constraint(self, max_cvar, method='historical'):
        """
        Maximize Sharpe ratio subject to CVaR constraint
        
        Parameters:
        -----------
        max_cvar : float
            Maximum allowed CVaR
        method : str
            'historical' or 'parametric' CVaR calculation
        """
        n_assets = len(self.tickers)
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # CVaR constraint
        if method == 'historical':
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_cvar - self.calculate_cvar_historical(x)
            })
        else:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_cvar - self.calculate_cvar_parametric(x)
            })
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Objective: Maximize Sharpe (minimize negative Sharpe)
        def objective(x):
            ret, std, _ = self.portfolio_performance(x)
            if std == 0:
                return 0
            return -(ret / std)  # Negative Sharpe for minimization
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.optimal_weights = result.x
        return result.x
    
    def generate_efficient_frontier(self, n_portfolios=100, method='historical'):
        """
        Generate efficient frontier with CVaR constraint
        
        Parameters:
        -----------
        n_portfolios : int
            Number of portfolios to generate
        method : str
            'historical' or 'parametric' for risk calculation
        """
        returns = []
        risks = []
        cvars = []
        vars = []
        sharpes = []
        weights_list = []
        
        # Generate random portfolios
        for _ in range(n_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= weights.sum()
            
            ret, risk, sharpe = self.portfolio_performance(weights)
            
            if method == 'historical':
                cvar = self.calculate_cvar_historical(weights)
                var = self.calculate_var_historical(weights)
            else:
                cvar = self.calculate_cvar_parametric(weights)
                var = self.calculate_var_parametric(weights)
            
            returns.append(ret)
            risks.append(risk)
            cvars.append(cvar)
            vars.append(var)
            sharpes.append(sharpe)
            weights_list.append(weights)
        
        frontier_df = pd.DataFrame({
            'Return': returns,
            'Risk': risks,
            'VaR': vars,
            'CVaR': cvars,
            'Sharpe': sharpes
        })
        
        # Add weights columns
        for i, ticker in enumerate(self.tickers):
            frontier_df[f'Weight_{ticker}'] = [w[i] for w in weights_list]
        
        return frontier_df
    
    def backtest(self, test_start_date, test_end_date, weights=None, rebalance_frequency='M'):
        """
        Backtest the portfolio strategy
        
        Parameters:
        -----------
        test_start_date : str
            Start date for backtesting
        test_end_date : str
            End date for backtesting
        weights : array, optional
            Portfolio weights (uses optimal_weights if not provided)
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
        """
        if weights is None:
            weights = self.optimal_weights
        
        # Fetch test data (robust selection of price column)
        raw = yf.download(self.tickers, start=test_start_date, end=test_end_date, auto_adjust=True)
        
        # Handle different data structures like in fetch_data
        if isinstance(raw.columns, pd.MultiIndex):
            if 'Close' in raw.columns.get_level_values(0):
                test_data = raw['Close']
            else:
                price_columns = [col for col in raw.columns.get_level_values(0) 
                                 if col in ['Close', 'Adj Close', 'Open', 'High', 'Low']]
                if price_columns:
                    test_data = raw[price_columns[0]]
                else:
                    raise ValueError("No suitable price columns found in the test data")
        else:
            # Single ticker case - ensure DataFrame with a price column
            if len(self.tickers) == 1:
                if 'Close' in raw.columns:
                    test_data = raw[['Close']]
                    test_data.columns = [self.tickers[0]]
                else:
                    test_data = raw.iloc[:, [0]]
                    test_data.columns = [self.tickers[0]]
            else:
                # Fallback: take first N columns for N tickers
                test_data = raw.iloc[:, :len(self.tickers)]
                # Try to rename columns to tickers when they're generic
                if not all(col in self.tickers for col in test_data.columns):
                    test_data.columns = [c if c in self.tickers else str(c) for c in test_data.columns]
        
        # Ensure column order matches self.tickers and filter to available tickers
        available_cols = [t for t in self.tickers if t in test_data.columns]
        if len(available_cols) == 0:
            raise ValueError("None of the requested tickers are present in the downloaded test data.")
        if len(available_cols) < len(self.tickers):
            missing = [t for t in self.tickers if t not in test_data.columns]
            print(f"Warning: Missing tickers in test period: {missing}. Proceeding with available tickers: {available_cols}")
        test_data = test_data[available_cols]
        
        test_returns = test_data.pct_change().dropna()
        
        # Align weights to available columns
        if weights is not None:
            weights_aligned = np.array([weights[self.tickers.index(t)] for t in available_cols])
            total = weights_aligned.sum()
            if total == 0:
                raise ValueError("Aligned weights sum to zero after filtering tickers.")
            weights = weights_aligned / total
        
        # Calculate portfolio returns
        portfolio_returns = (test_returns * weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility
        
        # Calculate maximum drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate VaR and CVaR on test data
        test_var = self.calculate_var_historical(weights, test_returns)
        test_cvar = self.calculate_cvar_historical(weights, test_returns)
        
        backtest_results = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'VaR (95%)': test_var,
            'CVaR (95%)': test_cvar,
            'Portfolio Returns': portfolio_returns,
            'Cumulative Returns': cumulative_returns
        }
        
        return backtest_results
    
    def plot_results(self, backtest_results=None):
        """
        Visualize portfolio optimization results
        
        Parameters:
        -----------
        backtest_results : dict, optional
            Results from backtesting
        """
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Portfolio Weights
        ax1 = plt.subplot(3, 3, 1)
        if self.optimal_weights is not None:
            ax1.bar(self.tickers, self.optimal_weights)
            ax1.set_title('Optimal Portfolio Weights')
            ax1.set_ylabel('Weight')
            ax1.set_xlabel('Asset')
            plt.xticks(rotation=45)
        
        # 2. Risk-Return Scatter
        ax2 = plt.subplot(3, 3, 2)
        frontier = self.generate_efficient_frontier(500)
        scatter = ax2.scatter(frontier['Risk'], frontier['Return'], 
                            c=frontier['Sharpe'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Sharpe Ratio')
        ax2.set_xlabel('Risk (Volatility)')
        ax2.set_ylabel('Expected Return')
        ax2.set_title('Risk-Return Efficient Frontier')
        
        # Mark optimal portfolio
        if self.optimal_weights is not None:
            opt_ret, opt_risk, _ = self.portfolio_performance(self.optimal_weights)
            ax2.scatter(opt_risk, opt_ret, color='red', s=200, marker='*', 
                      edgecolors='black', linewidth=2, label='Optimal Portfolio')
            ax2.legend()
        
        # 3. CVaR vs VaR
        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(frontier['VaR'], frontier['CVaR'], alpha=0.6)
        ax3.set_xlabel('VaR (95%)')
        ax3.set_ylabel('CVaR (95%)')
        ax3.set_title('VaR vs CVaR Trade-off')
        
        # 4. Historical Returns Distribution
        ax4 = plt.subplot(3, 3, 4)
        if self.optimal_weights is not None:
            portfolio_returns = (self.returns * self.optimal_weights).sum(axis=1)
            ax4.hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
            
            # Mark VaR and CVaR
            var_line = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
            ax4.axvline(-self.calculate_var_historical(self.optimal_weights), 
                       color='red', linestyle='--', label=f'VaR ({self.confidence_level*100}%)')
            ax4.axvline(-self.calculate_cvar_historical(self.optimal_weights), 
                       color='orange', linestyle='--', label=f'CVaR ({self.confidence_level*100}%)')
            ax4.set_xlabel('Daily Returns')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Portfolio Returns Distribution')
            ax4.legend()
        
        # 5. Correlation Matrix
        ax5 = plt.subplot(3, 3, 5)
        sns.heatmap(self.returns.corr(), annot=True, cmap='coolwarm', center=0, ax=ax5)
        ax5.set_title('Asset Correlation Matrix')
        
        # 6. Cumulative Returns (if backtest results available)
        if backtest_results is not None:
            ax6 = plt.subplot(3, 3, 6)
            backtest_results['Cumulative Returns'].plot(ax=ax6)
            ax6.set_xlabel('Date')
            ax6.set_ylabel('Cumulative Return')
            ax6.set_title('Backtest Cumulative Returns')
            ax6.grid(True, alpha=0.3)
            
            # 7. Drawdown
            ax7 = plt.subplot(3, 3, 7)
            rolling_max = backtest_results['Cumulative Returns'].expanding().max()
            drawdown = (backtest_results['Cumulative Returns'] - rolling_max) / rolling_max
            drawdown.plot(ax=ax7, color='red')
            ax7.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
            ax7.set_xlabel('Date')
            ax7.set_ylabel('Drawdown')
            ax7.set_title('Portfolio Drawdown')
            ax7.grid(True, alpha=0.3)
            
            # 8. Rolling Volatility
            ax8 = plt.subplot(3, 3, 8)
            rolling_vol = backtest_results['Portfolio Returns'].rolling(window=30).std() * np.sqrt(252)
            rolling_vol.plot(ax=ax8)
            ax8.set_xlabel('Date')
            ax8.set_ylabel('Annualized Volatility')
            ax8.set_title('30-Day Rolling Volatility')
            ax8.grid(True, alpha=0.3)
        
        # 9. Risk Decomposition
        ax9 = plt.subplot(3, 3, 9)
        if self.optimal_weights is not None:
            # Calculate risk contribution
            portfolio_vol = np.sqrt(np.dot(self.optimal_weights.T, 
                                          np.dot(self.cov_matrix, self.optimal_weights)))
            marginal_contrib = np.dot(self.cov_matrix, self.optimal_weights) / portfolio_vol
            contrib = self.optimal_weights * marginal_contrib
            contrib_pct = contrib / contrib.sum() * 100
            
            ax9.bar(self.tickers, contrib_pct)
            ax9.set_xlabel('Asset')
            ax9.set_ylabel('Risk Contribution (%)')
            ax9.set_title('Risk Contribution by Asset')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_optimization_summary(self):
        """Generate comprehensive optimization summary"""
        if self.optimal_weights is None:
            return "No optimization has been performed yet."
        
        summary = {}
        
        # Portfolio composition
        summary['Portfolio Weights'] = dict(zip(self.tickers, self.optimal_weights))
        
        # Performance metrics
        ret, risk, sharpe = self.portfolio_performance(self.optimal_weights)
        summary['Expected Annual Return'] = f"{ret*100:.2f}%"
        summary['Annual Volatility'] = f"{risk*100:.2f}%"
        summary['Sharpe Ratio'] = f"{sharpe:.3f}"
        
        # Risk metrics
        summary['VaR (95%, Historical)'] = f"{self.calculate_var_historical(self.optimal_weights)*100:.2f}%"
        summary['CVaR (95%, Historical)'] = f"{self.calculate_cvar_historical(self.optimal_weights)*100:.2f}%"
        summary['VaR (95%, Parametric)'] = f"{self.calculate_var_parametric(self.optimal_weights)*100:.2f}%"
        summary['CVaR (95%, Parametric)'] = f"{self.calculate_cvar_parametric(self.optimal_weights)*100:.2f}%"
        
        return summary
    



#Main functions executions
if __name__ == "__main__":
    # Define portfolio parameters
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JPM', 'XOM', 'JNJ', 'V']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # Initialize optimizer
    print("="*60)
    print("Classical Portfolio Optimization with VaR/CVaR")
    print("="*60)
    
    optimizer = PortfolioOptimizer(tickers, start_date, end_date, confidence_level=0.95)
    
    # Fetch data
    optimizer.fetch_data()
    
    # Calculate statistics
    optimizer.calculate_statistics()
    
    print("\nExpected Annual Returns:")
    for ticker, ret in zip(optimizer.tickers, optimizer.expected_returns):
        print(f"{ticker}: {ret*100:.2f}%")
    
    # Optimize portfolio - Minimize CVaR
    print("\n" + "="*60)
    print("Optimization Method 1: Minimize CVaR")
    print("="*60)
    
    weights_min_cvar = optimizer.optimize_min_cvar(method='historical')
    summary_min_cvar = optimizer.get_optimization_summary()
    
    print("\nOptimal Weights:")
    for ticker, weight in zip(optimizer.tickers, weights_min_cvar):
        print(f"{ticker}: {weight*100:.2f}%")
    
    print("\nPortfolio Metrics:")
    for key, value in summary_min_cvar.items():
        if key != 'Portfolio Weights':
            print(f"{key}: {value}")
    
    # Optimize portfolio - Mean-CVaR
    print("\n" + "="*60)
    print("Optimization Method 2: Mean-CVaR Trade-off")
    print("="*60)
    
    weights_mean_cvar = optimizer.optimize_mean_cvar(lambda_param=0.5, method='historical')
    summary_mean_cvar = optimizer.get_optimization_summary()
    
    print("\nOptimal Weights:")
    for ticker, weight in zip(optimizer.tickers, weights_mean_cvar):
        print(f"{ticker}: {weight*100:.2f}%")
    
    print("\nPortfolio Metrics:")
    for key, value in summary_mean_cvar.items():
        if key != 'Portfolio Weights':
            print(f"{key}: {value}")
    
    # Optimize portfolio - Maximize Sharpe with CVaR constraint
    print("\n" + "="*60)
    print("Optimization Method 3: Maximize Sharpe with CVaR Constraint")
    print("="*60)
    
    max_cvar = 0.02  # Maximum 2% CVaR
    weights_sharpe_cvar = optimizer.optimize_sharpe_with_cvar_constraint(max_cvar, method='historical')
    summary_sharpe_cvar = optimizer.get_optimization_summary()
    
    print("\nOptimal Weights:")
    for ticker, weight in zip(optimizer.tickers, weights_sharpe_cvar):
        print(f"{ticker}: {weight*100:.2f}%")
    
    print("\nPortfolio Metrics:")
    for key, value in summary_sharpe_cvar.items():
        if key != 'Portfolio Weights':
            print(f"{key}: {value}")
    
    # Backtest the strategy
    print("\n" + "="*60)
    print("Backtesting Results (2024 Data)")
    print("="*60)
    
    backtest_results = optimizer.backtest('2024-01-01', '2024-06-30', weights_mean_cvar)
    
    print(f"Total Return: {backtest_results['Total Return']*100:.2f}%")
    print(f"Annual Return: {backtest_results['Annual Return']*100:.2f}%")
    print(f"Volatility: {backtest_results['Volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {backtest_results['Sharpe Ratio']:.3f}")
    print(f"Max Drawdown: {backtest_results['Max Drawdown']*100:.2f}%")
    print(f"VaR (95%): {backtest_results['VaR (95%)']*100:.2f}%")
    print(f"CVaR (95%): {backtest_results['CVaR (95%)']*100:.2f}%")
    
    # Export parameters for quantum computing
    print("\n" + "="*60)
    print("Exporting Parameters for Quantum Computing")
    print("="*60)
    
  
    # Visualize results
    print("\n" + "="*60)
    print("Generating Visualization...")
    print("="*60)
    
    optimizer.plot_results(backtest_results)
    
    print("\nOptimization complete! Ready for quantum computing integration.")
    print("The quantum_params dictionary contains all necessary data for QUBO formulation.")