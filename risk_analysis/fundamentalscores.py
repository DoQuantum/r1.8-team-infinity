import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class FundamentalScoresCalculator:
    """
    Simplified class to calculate only fundamental scores
    """
    
    def __init__(self, symbols):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.stock_data = {}
        self.scores = {}
        
    def fetch_essential_data(self):
        """
        Fetch only essential data needed for scoring
        """
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                balance_sheet = ticker.balance_sheet
                
                self.stock_data[symbol] = {
                    'info': info,
                    'balance_sheet': balance_sheet
                }
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
    
    def extract_key_metrics(self, symbol):
        """
        Extract only the metrics needed for scoring
        """
        info = self.stock_data[symbol]['info']
        balance_sheet = self.stock_data[symbol]['balance_sheet']
        
        metrics = {}
        
        # Profitability metrics
        metrics['roe'] = info.get('returnOnEquity', np.nan)
        metrics['roa'] = info.get('returnOnAssets', np.nan)
        metrics['profit_margin'] = info.get('profitMargins', np.nan)
        metrics['operating_margin'] = info.get('operatingMargins', np.nan)
        metrics['gross_margin'] = info.get('grossMargins', np.nan)
        
        # Valuation metrics
        metrics['pe_ratio'] = info.get('trailingPE', np.nan)
        metrics['pb_ratio'] = info.get('priceToBook', np.nan)
        metrics['ps_ratio'] = info.get('priceToSalesTrailing12Months', np.nan)
        metrics['peg_ratio'] = info.get('pegRatio', np.nan)
        metrics['ev_ebitda'] = info.get('enterpriseToEbitda', np.nan)
        
        # Liquidity metrics
        if not balance_sheet.empty and len(balance_sheet.columns) > 0:
            latest_bs = balance_sheet.iloc[:, 0]
            current_assets = latest_bs.get('Current Assets', np.nan)
            current_liabilities = latest_bs.get('Current Liabilities', np.nan)
            cash = latest_bs.get('Cash And Cash Equivalents', np.nan)
            inventory = latest_bs.get('Inventory', np.nan)
            
            if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities != 0:
                metrics['current_ratio'] = current_assets / current_liabilities
            else:
                metrics['current_ratio'] = np.nan
                
            if pd.notna(current_assets) and pd.notna(current_liabilities) and pd.notna(inventory) and current_liabilities != 0:
                metrics['quick_ratio'] = (current_assets - inventory) / current_liabilities
            else:
                metrics['quick_ratio'] = np.nan
                
            if pd.notna(cash) and pd.notna(current_liabilities) and current_liabilities != 0:
                metrics['cash_ratio'] = cash / current_liabilities
            else:
                metrics['cash_ratio'] = np.nan
        else:
            metrics['current_ratio'] = np.nan
            metrics['quick_ratio'] = np.nan
            metrics['cash_ratio'] = np.nan
        
        # Leverage metrics
        metrics['debt_to_equity'] = info.get('debtToEquity', np.nan)
        
        if not balance_sheet.empty and len(balance_sheet.columns) > 0:
            latest_bs = balance_sheet.iloc[:, 0]
            total_debt = latest_bs.get('Total Debt', np.nan)
            total_equity = latest_bs.get('Stockholders Equity', np.nan)
            total_assets = latest_bs.get('Total Assets', np.nan)
            
            if pd.notna(total_debt) and pd.notna(total_assets) and total_assets != 0:
                metrics['debt_to_assets'] = total_debt / total_assets
            else:
                metrics['debt_to_assets'] = np.nan
                
            if pd.notna(total_equity) and pd.notna(total_assets) and total_assets != 0:
                metrics['equity_ratio'] = total_equity / total_assets
            else:
                metrics['equity_ratio'] = np.nan
        else:
            metrics['debt_to_assets'] = np.nan
            metrics['equity_ratio'] = np.nan
        
        # Growth metrics
        metrics['revenue_growth'] = info.get('revenueGrowth', np.nan)
        metrics['earnings_growth'] = info.get('earningsGrowth', np.nan)
        metrics['earnings_quarterly_growth'] = info.get('earningsQuarterlyGrowth', np.nan)
        
        # Dividend metrics
        metrics['dividend_yield'] = info.get('dividendYield', np.nan)
        metrics['payout_ratio'] = info.get('payoutRatio', np.nan)
        
        return metrics
    
    def calculate_scores(self):
        """
        Calculate fundamental scores for all symbols
        """
        self.fetch_essential_data()
        
        # Extract metrics for all symbols
        all_metrics = {}
        for symbol in self.symbols:
            if symbol in self.stock_data:
                all_metrics[symbol] = self.extract_key_metrics(symbol)
        
        if not all_metrics:
            return {}
        
        # Create DataFrame for normalization
        df = pd.DataFrame(all_metrics).T
        
        # Define scoring configuration (1 = higher is better, -1 = lower is better)
        scoring_config = {
            # Profitability (higher is better)
            'roe': 1, 'roa': 1, 'profit_margin': 1, 'operating_margin': 1, 'gross_margin': 1,
            # Valuation (lower is better)
            'pe_ratio': -1, 'pb_ratio': -1, 'ps_ratio': -1, 'peg_ratio': -1, 'ev_ebitda': -1,
            # Liquidity (higher is better)
            'current_ratio': 1, 'quick_ratio': 1, 'cash_ratio': 1,
            # Leverage (lower debt is better, higher equity is better)
            'debt_to_equity': -1, 'debt_to_assets': -1, 'equity_ratio': 1,
            # Growth (higher is better)
            'revenue_growth': 1, 'earnings_growth': 1, 'earnings_quarterly_growth': 1,
            # Dividend (higher is better)
            'dividend_yield': 1, 'payout_ratio': 1
        }
        
        # Calculate scores for each symbol
        for symbol in self.symbols:
            if symbol in all_metrics:
                scores = {}
                
                # Category groupings
                categories = {
                    'profitability': ['roe', 'roa', 'profit_margin', 'operating_margin', 'gross_margin'],
                    'valuation': ['pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio', 'ev_ebitda'],
                    'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
                    'leverage': ['debt_to_equity', 'debt_to_assets', 'equity_ratio'],
                    'growth': ['revenue_growth', 'earnings_growth', 'earnings_quarterly_growth'],
                    'dividend': ['dividend_yield', 'payout_ratio']
                }
                
                # Calculate category scores
                for category, metrics in categories.items():
                    category_scores = []
                    
                    for metric in metrics:
                        if metric in df.columns and metric in scoring_config:
                            value = df.loc[symbol, metric]
                            if pd.notna(value):
                                # Min-max normalization
                                col_values = df[metric].dropna()
                                if len(col_values) > 1:
                                    min_val, max_val = col_values.min(), col_values.max()
                                    if max_val != min_val:
                                        normalized = (value - min_val) / (max_val - min_val)
                                        # Apply direction
                                        if scoring_config[metric] == -1:
                                            normalized = 1 - normalized
                                        category_scores.append(normalized)
                    
                    scores[f'{category}_score'] = np.mean(category_scores) if category_scores else 0.5
                
                # Overall score (weighted average)
                weights = {
                    'profitability_score': 0.25,
                    'valuation_score': 0.20,
                    'liquidity_score': 0.10,
                    'leverage_score': 0.15,
                    'growth_score': 0.20,
                    'dividend_score': 0.10
                }
                
                overall_score = sum(scores[key] * weights[key] for key in weights.keys() if key in scores)
                scores['overall_fundamental_score'] = overall_score
                
                self.scores[symbol] = scores
        
        return self.scores
    
    def get_scores_summary(self):
        """
        Return a clean summary of scores
        """
        if not self.scores:
            self.calculate_scores()
        
        summary = {}
        for symbol in self.symbols:
            if symbol in self.scores:
                summary[symbol] = {
                    'Overall Fundamental Score': round(self.scores[symbol]['overall_fundamental_score'], 3),
                    'Profitability Score': round(self.scores[symbol]['profitability_score'], 3),
                    'Valuation Score': round(self.scores[symbol]['valuation_score'], 3),
                    'Liquidity Score': round(self.scores[symbol]['liquidity_score'], 3),
                    'Leverage Score': round(self.scores[symbol]['leverage_score'], 3),
                    'Growth Score': round(self.scores[symbol]['growth_score'], 3),
                    'Dividend Score': round(self.scores[symbol]['dividend_score'], 3)
                }
        
        return summary
    
    def print_scores(self):
        """
        Print scores in a clean format
        """
        summary = self.get_scores_summary()
        
        for symbol, scores in summary.items():
            print(f"\n{symbol} - Fundamental Scores:")
            print("-" * 30)
            for score_name, score_value in scores.items():
                print(f"{score_name}: {score_value}")

# Simple usage function
def get_fundamental_scores(symbols):
    """
    Simple function to get fundamental scores for given symbols
    
    Args:
        symbols: Single symbol (str) or list of symbols
    
    Returns:
        Dictionary with scores for each symbol
    """
    calculator = FundamentalScoresCalculator(symbols)
    return calculator.get_scores_summary()

 
if __name__ == "__main__":
    print("Multiple Stocks Analysis")
    
    
    multi_scores = get_fundamental_scores(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'JPM'])
    
    for symbol, symbol_scores in multi_scores.items():
        print(f"\n{symbol}:")
        for score_name, score_value in symbol_scores.items():
            print(f"  {score_name}: {score_value}")