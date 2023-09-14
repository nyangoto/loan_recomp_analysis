# Loan Portfolio Analysis Dashboard

![Dashboard Screenshot](screenshot.png)

## Overview

This repository contains a Python-based loan portfolio analysis dashboard that allows financial analysts and executives to visualize and explore various metrics related to loan performance, portfolio overview, and trends over time. The dashboard is built using Streamlit, Plotly, and Pandas libraries.

## Features

- **Interest Variances Visualization**: Visualize main, penal, and total interest variances using box plots.
- **Loan Portfolio Overview**: View the composition of the loan portfolio by types of loans, currencies, and more.
- **Loan Maturity Distribution**: Understand the distribution of loan maturities.
- **Loan Repayment Patterns**: Analyze loan repayment trends over time.
- **Loan Aging Analysis**: Explore loan aging patterns and overdue days.
- **Loan Components Over Time**: Visualize the changes in principal and interest components over time.
- **Loan Performance by Customer**: Identify top-performing and bottom-performing customers.
- **Loan Delinquency Rate Over Time**: Track the delinquency rate of loans over time.
- **Accruals vs. Settled Amounts**: Compare accrued interest with settled interest amounts.
- **Loan Portfolio Concentration**: Check if the loan portfolio is concentrated in specific product descriptions.
- **Effective vs. System-Computed Interest**: Compare effective interest with system-computed interest.

## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/loan-portfolio-analysis.git

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
4. Open a web browser and navigate to the provided URL (usually http://localhost:8501) to access the dashboard.

Data Sources
The project uses three CSV data sources:

Accruals_Table.csv: Contains data related to accruals and interest amounts.
Loan_Listing_Table.csv: Represents the loan book with customer details and loan information.
Loan_Repayment_Schedule.csv: Provides repayment schedule data for loans.
License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributors
Patrick Orangi

## Acknowledgments
Special thanks to the Streamlit, Plotly, and Pandas communities for their fantastic tools and resources.

Feel free to contribute to the project by opening issues, suggesting improvements, or submitting pull requests.

Happy analyzing!
