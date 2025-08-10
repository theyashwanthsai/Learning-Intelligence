import os
from alpha_vantage.fundamentaldata import FundamentalData
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_fundamental_data(ticker: str) -> str:
    """Tool to get fundamental data for a given company stock ticker, including income statement and company overview."""
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        return "Error: ALPHA_VANTAGE_API_KEY not found in environment variables."
        
    fd = FundamentalData(key, output_format='json')
    
    try:
        # The method returns a tuple: (data, metadata). We only need the data.
        # The data itself is the list of reports.
        income_statement_reports, _ = fd.get_income_statement_annual(ticker)
        overview = fd.get_company_overview(ticker)
        print(overview)
        print(income_statement_reports)
        # Directly access the first item in the list, which is the most recent report.
        annual_report = income_statement_reports[0] 
        
        # Using .get() is a good practice to avoid errors if a key is missing.
        net_income = annual_report.get('netIncome', 'N/A')
        total_revenue = annual_report.get('totalRevenue', 'N/A')
        fiscal_date = annual_report.get('fiscalDateEnding', 'N/A')
        
        # Check if the overview data is available
        description = overview.get('Description', 'No description available.')

        return (f"Company Overview: {description}\n\n"
                f"Most Recent Annual Report ({fiscal_date}):\n"
                f"Net Income: {net_income}\n"
                f"Total Revenue: {total_revenue}")

    except (IndexError, TypeError):
        # This handles cases where no report data is returned
        return f"Error: No annual report data found for {ticker}."
    except Exception as e:
        return f"Error fetching fundamental data: {e}"

# Example usage:
print(get_fundamental_data("TSLA"))