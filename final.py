import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Import Data and Data Preparation

# Import CSV files
loan_listing_df = pd.read_csv("Loan_Listing_Table.csv")
accruals_df = pd.read_csv("Accruals_Table.csv")
repayment_schedule_df = pd.read_csv("Loan_Repayment_schedule_table.csv")

# Data cleaning and type conversion if needed
# For example, convert date columns to datetime format if not already done:
loan_listing_df['Value_Date'] = pd.to_datetime(loan_listing_df['Value_Date'])
accruals_df['Value_Date'] = pd.to_datetime(accruals_df['Value_Date'])
repayment_schedule_df['Schedule_Due_Date'] = pd.to_datetime(repayment_schedule_df['Schedule_Due_Date'])

# Step 2: Merge Loan Listing Data

# Import the opening loan book (31.12.2021)
#opening_loan_book_df = pd.read_csv("Loan_Listing_Table.csv")  # Replace with your actual file name.

# Combine the monthly loan data and the opening loan book into one DataFrame
#loan_listing_df = loan_listing_df

# Select the relevant fields from the consolidated loan data
#selected_fields = [
#    'Account_Number',
#    'Customer_Name',
#    'Product_Description',
#    'Currency',
#    'Value_Date',
#    'Maturity_Date',
#    'Interest_Rate',
#    'Amount_Financed',
#    'Principal_os_amount',
#    'Principal_Overdue_Amount',
#    'Interest_Overdue_Amount',
#    'Overdue_Days',
#    'Branch_Code',
#    'PRINCIPAL_INSTALLMENTS',
#    'CR_ACC',
#]

#loan_listing_df = loan_listing_df[selected_fields]
#print(loan_listing_df.columns)

# Step 3: Adjust Principal Outstanding for New Loans

# Convert Value_Date to timestamp type
loan_listing_df['Value_Date'] = pd.to_datetime(loan_listing_df['Value_Date'])

# Filter for loans created after January 1, 2022
cutoff_date = pd.Timestamp('2022-01-01')
new_loans_df = loan_listing_df[loan_listing_df['Value_Date'] >= cutoff_date]

# Use a conditional formula to adjust Principal Outstanding for new loans
loan_listing_df['Principal_os_amount'] = np.where(
    loan_listing_df['Value_Date'] >= '2022-01-01',
    loan_listing_df['Amount_Financed'],
    loan_listing_df['Principal_os_amount']
)

# Determine the Accrued Interest amount
loan_listing_df['Interest_Overdue_Amount'] = np.where(
    loan_listing_df['Value_Date'] >= '2022-01-01',
    0,
    loan_listing_df['Interest_Overdue_Amount']
)

# Remove null records
loan_listing_df.dropna(subset=['Principal_os_amount'], inplace=True)

#print(loan_listing_df.head(8))

# Step 4: Generate Daily Records for Loans

# Define constant variables
year_start = datetime(2022, 1, 1)
year_end = datetime(2022, 12, 31)

# Calculate the date differences
loan_listing_df['Date_Diff'] = (pd.to_datetime(loan_listing_df['Value_Date']) - year_start).dt.days

# Create a new field for Accrual Start Date
# loan_listing_df['Accrual_Start_Date'] = np.where(
#    loan_listing_df['Value_Date'] >= '2022-01-01',
#    year_start,
#    loan_listing_df['Value_Date']
# )

# Create a new field for Accrual Start Date based on the condition
loan_listing_df['Accrual_Start_Date'] = loan_listing_df.apply(
    lambda row: row['Value_Date'] if row['Date_Diff'] > 1 else year_start,
    axis=1
)

# Generate daily records
daily_records = []
for _, row in loan_listing_df.iterrows():
    loan_account = row['Account_Number']
    accrual_start_date = row['Accrual_Start_Date']
    
    while accrual_start_date.year == year_start.year:
        daily_records.append({
            'Account_Number': loan_account,
            'Accrual_Date': accrual_start_date
        })
        accrual_start_date += pd.DateOffset(days=1)

# Create a DataFrame from the daily records
daily_records_df = pd.DataFrame(daily_records)

# Merge 'Currency' information into 'daily_records_df'
daily_records_df = daily_records_df.merge(loan_listing_df[['Account_Number', 'Currency']], on='Account_Number', how='left')

# Define BaseDays based on Loan Currency
daily_records_df['BaseDays'] = np.where(
    daily_records_df['Currency'] == 'KES',
    365,
    360
)

#print(daily_records_df.head(8))

# Step 5: Import and Prepare Accruals Data
# Create separate fields for Principal Liquidation, Main Interest Liquidation, and Accrued Interest Liquidation
accruals_df['Principal_Liquidation_Amount'] = np.where(
    accruals_df['Amount_Tag'] == 'Principal Liquidation',
    accruals_df['LCY_Amount'],
    0
)

accruals_df['Main_Interest_Liquidation_Amount'] = np.where(
    accruals_df['Amount_Tag'] == 'Main Interest Liquidation',
    accruals_df['LCY_Amount'],
    0
)

accruals_df['Accrued_Interest_Liquidation_Amount'] = np.where(
    accruals_df['Amount_Tag'] == 'Accrued Interest Liquidation',
    accruals_df['LCY_Amount'],
    0
)

# Select relevant fields
selected_accruals_fields = [
    'Account_Number',
    'Event_Code',
    'Value_Date',
    'Amount_Settled',
    'Amount_Tag',
    'LCY_Amount',
    'Principal_Liquidation_Amount',
    'Main_Interest_Liquidation_Amount',
    'Accrued_Interest_Liquidation_Amount',
]

accruals_df = accruals_df[selected_accruals_fields]
#print(loan_listing_df.columns)

# Step 6: Bring in System Liquidation Data
# Merge the consolidated loan data with the accruals data
merged_loan_accruals_df = pd.merge(loan_listing_df, accruals_df, on=['Account_Number', 'Value_Date'], how='left')

# Assuming daily_records_df contains 'Account_Number' and 'BaseDays' columns
# Merge daily_records_df into merged_loan_accruals_df based on 'Account_Number'
merged_loan_accruals_df = merged_loan_accruals_df.merge(daily_records_df[['Account_Number', 'BaseDays']], on='Account_Number', how='left')
#print(merged_loan_accruals_df.columns)

# Now, merged_loan_accruals_df contains the loan data with the added accruals information.


# Step 7: Bring in Repayment schedule data
# Assuming you have the following DataFrames available:
# repayment_schedule_df: Repayment schedule data (from Step 8)

# Filter rows for Interest Due Dates
interest_due_df = repayment_schedule_df[repayment_schedule_df['Component_Name'] == 'Interest']

# Filter rows for Principal Due Dates
principal_due_df = repayment_schedule_df[repayment_schedule_df['Component_Name'] == 'Principal']

# Create a flag field for scheduled repayment dates (repayment_flag)
#repayment_schedule_df['repayment_flag'] = 0
#repayment_schedule_df.loc[~repayment_schedule_df['Schedule_Due_Date'].isna(), 'repayment_flag'] = 1

# Assuming merged_loan_accruals_df is the DataFrame from Step 6
# You can join it with the filtered interest and principal due DataFrames

# Join with interest_due_df
merged_loan_accruals_df = pd.merge(merged_loan_accruals_df, interest_due_df, 
                                    left_on=['Account_Number', 'Value_Date'], 
                                    right_on=['Account_Number', 'Schedule_Due_Date'], 
                                    how='left', suffixes=('', '_Interest'))

# Join with principal_due_df
merged_loan_accruals_df = pd.merge(merged_loan_accruals_df, principal_due_df, 
                                    left_on=['Account_Number', 'Value_Date'], 
                                    right_on=['Account_Number', 'Schedule_Due_Date'],
                                    how='left', suffixes=('', '_Principal'))

# Now, merged_loan_accruals_df contains the filtered interest and principal due dates.
# You can use these fields in Step 8 for calculations.

#print(merged_loan_accruals_df.columns)


# Step 8: PwC Interest Computation

# Assuming you have the following DataFrames available:
# merged_loan_accruals_df: Merged loan data with accruals (from Step 6)
# repayment_schedule_df: Repayment schedule data (from Step 8)

# Concatenate the DataFrames vertically
#merged_loan_accruals_df = pd.merge(merged_loan_accruals_df, loan_listing_df, on='Account_Number', how='inner')
#selected_rows = merged_loan_accruals_df['Principal_Liquidation_Amount'].head(8)
#print(merged_loan_accruals_df.columns)

# Initialize 'INT_ACCRUED', 'DAILY_INT_AMT', and 'REPAYMENT_FLAG' columns with NaN values
merged_loan_accruals_df['INT_ACCRUED'] = np.nan
merged_loan_accruals_df['DAILY_INT_AMT'] = np.nan
merged_loan_accruals_df['REPAYMENT_FLAG'] = np.nan

# Loop through each unique loan account
unique_accounts = merged_loan_accruals_df['Account_Number'].unique()
for account_number in unique_accounts:
    # Select data for the current loan account
    loan_data = merged_loan_accruals_df[merged_loan_accruals_df['Account_Number'] == account_number]
    loan_data = loan_data.sort_values(by='Value_Date')  # Sort by Value_Date if needed
    
    #principal_installments = loan_listing_df.loc[loan_listing_df['Account_Number'] == account_number, 'PRINCIPAL_INSTALLMENTS'].values[0]
    
    # Initialize an empty list for INT_ACCRUED
    int_accrued_list = []

    # Initialize an empty list for DAILY_INT_AMT
    daily_int_amount_list = []
    
    # Initialize an empty list for repayment_flag
    repayment_flag_list = []

    # Loop through loan_data
    for index, row in loan_data.iterrows():
        # Update Principal_os_amount
        if index == 0:
            principal_os_amount = row['Principal_os_amount']
        else:
            principal_os_amount = row['Principal_os_amount'] - row['Principal_Liquidation_Amount']

        # Update Principal_Overdue_Amount
        if index == 0 or pd.isnull(row['Principal_Overdue_Amount']):
            principal_overdue_amount = row['PRINCIPAL_INSTALLMENTS'] - row['Principal_Liquidation_Amount']
        else:
            principal_overdue_amount = row['Principal_Overdue_Amount'] + row['PRINCIPAL_INSTALLMENTS'] - row['Principal_Liquidation_Amount']

        # Calculate DAILY_INT_AMOUNT
        if principal_os_amount >= 0:
            daily_int_amount = (principal_os_amount * row['Interest_Rate'] / (row['BaseDays'] * 100)) * 1
        else:
            daily_int_amount = 0

        # Append to the list
        daily_int_amount_list.append(daily_int_amount)
        
        # Set REPAYMENT_FLAG
        repayment_flag = 1 if not pd.isnull(row['Schedule_Due_Date']) else 0

        # Append repayment_flag to the list
        repayment_flag_list.append(repayment_flag)

        # Calculate int_accrued
        if index == 0 or repayment_flag == 1 or pd.isnull(repayment_flag):
            int_accrued = daily_int_amount
        else:
            # Check if int_accrued_list is not empty
            if int_accrued_list:
                int_accrued = daily_int_amount + int_accrued_list[-1]
            else:
                int_accrued = daily_int_amount

        # Append int_accrued to INT_ACCRUED list
        int_accrued_list.append(int_accrued)

    # Update the DataFrame with the lists
    merged_loan_accruals_df.loc[loan_data.index, 'INT_ACCRUED'] = int_accrued_list
    merged_loan_accruals_df.loc[loan_data.index, 'DAILY_INT_AMT'] = daily_int_amount_list
    merged_loan_accruals_df.loc[loan_data.index, 'REPAYMENT_FLAG'] = repayment_flag_list

#print(merged_loan_accruals_df.columns)

# Step 9: Interest Suspension and in Duplum Rule &  10:Compute DR to Int Suspense A/C
# Assuming you have the following DataFrames available:
# merged_loan_accruals_df: Merged loan data with accruals and repayment_flag (from Step 7)
# repayment_schedule_df: Repayment schedule data (from Step 8)

# join computed interest df & merged accruals df 
#ef_interest_df=pd.concat([computed_interest_df, merged_loan_accruals_df], ignore_index=True)

# Initialize new column names# Initialize 'INT_ACCRUED', 'DAILY_INT_AMT', and 'REPAYMENT_FLAG' columns with NaN values
merged_loan_accruals_df['LAST_REPAYMENT_DATE'] = np.nan
merged_loan_accruals_df['INT_SUSPENSION_FLAG'] = np.nan
merged_loan_accruals_df['DR_TO_SUSPENDED_INT'] = np.nan
merged_loan_accruals_df['DAYS_PNL_COMP'] = np.nan
merged_loan_accruals_df['EFFECTIVE_INTEREST'] = np.nan
merged_loan_accruals_df['PENAL_INTEREST'] = np.nan


# Create empty lists to store intermediate values
lrd_list = []
isf_list = []
dtsi_list = []
dpc_list = []
ei_list = []
pi_list = []


for index, row in merged_loan_accruals_df.iterrows():
    # Calculate OVERDUE INTEREST
    if pd.isnull(row['Interest_Overdue_Amount']):
        interest_overdue = row['Interest_Overdue_Amount']
    elif row['REPAYMENT_FLAG'] == 1:
        interest_overdue = row['INT_ACCRUED'] + row['Interest_Overdue_Amount'] - row['Main_Interest_Liquidation_Amount'] - row['Accrued_Interest_Liquidation_Amount']
    else:
        interest_overdue = row['Interest_Overdue_Amount'] - row['Main_Interest_Liquidation_Amount'] - row['Accrued_Interest_Liquidation_Amount']

    # Calculate last initial repayment date
    last_initial_repayment_date = None
    if pd.isnull(row['LAST_REPAYMENT_DATE']):
        last_initial_repayment_date = row['LAST_REPAYMENT_DATE']
    elif row['REPAYMENT_FLAG'] == 1 and row['Interest_Overdue_Amount'] <= row['INT_ACCRUED'] and row['Principal_Overdue_Amount'] <= row['PRINCIPAL_INSTALLMENTS']:
        last_initial_repayment_date = row['Schedule_Due_Date']

    lrd_list.append(last_initial_repayment_date)

    # Calculate INT_SUSPENSION_FLAG
    int_suspension_flag = 1 if row['Overdue_Days'] >= 90 else 0
    isf_list.append(int_suspension_flag)

    # Calculate DR_TO_SUSPENDED_INT
    dr_to_suspended_int = None
    if pd.isnull(row['INT_SUSPENSION_FLAG']):
        dr_to_suspended_int = 0
    elif row['INT_SUSPENSION_FLAG'] == 0 and int_suspension_flag == 1:
        dr_to_suspended_int = row['Main_Interest_Liquidation_Amount'] + row['Accrued_Interest_Liquidation_Amount'] - row['Interest_Overdue_Amount'] - row['INT_ACCRUED']

    dtsi_list.append(dr_to_suspended_int)

    # Calculate DAYS_PNL_COMP
    days_pnl_comp = None
    if pd.isnull(row['Overdue_Days']):
        days_pnl_comp = row['Overdue_Days']
    else:
        days_pnl_comp = row['Overdue_Days'] - merged_loan_accruals_df.at[index - 1, 'Overdue_Days'] if index > 0 else row['Overdue_Days']

    dpc_list.append(days_pnl_comp)

    # Calculate Effective Interest
    effective_interest = None
    if row['INT_SUSPENSION_FLAG'] == 0:
        effective_interest = row['DAILY_INT_AMT']
    elif row['INT_SUSPENSION_FLAG'] == 1 and dr_to_suspended_int > 0:
        effective_interest = dr_to_suspended_int
    else:
        effective_interest = 0

    ei_list.append(effective_interest)

    # Calculate Penal Interest
    penal_interest = ((row['Interest_Overdue_Amount'] + row['Principal_Overdue_Amount']) * 9) / (100 * row['BaseDays']) if (
            row['Interest_Overdue_Amount'] + row['Principal_Overdue_Amount']) >= 0 else 0

    pi_list.append(penal_interest)

# Assign the lists to the DataFrame
merged_loan_accruals_df['LAST_REPAYMENT_DATE'] = lrd_list
merged_loan_accruals_df['INT_SUSPENSION_FLAG'] = isf_list
merged_loan_accruals_df['DR_TO_SUSPENDED_INT'] = dtsi_list
merged_loan_accruals_df['DAYS_PNL_COMP'] = dpc_list
merged_loan_accruals_df['EFFECTIVE_INTEREST'] = ei_list
merged_loan_accruals_df['PENAL_INTEREST'] = pi_list


#print(merged_loan_accruals_df.columns)

# Step 11: Prepare Comparison Data

# Summarize the accrued interest amounts per ACCOUNT_NUMBER and AMOUNT_TAG
# Assuming you have the following DataFrames available:
# merged_loan_accruals_df: Merged loan data with computed fields (from Step 10)

# Identify interest in suspense
merged_loan_accruals_df['INT_SUSPENSE_FLAG'] = merged_loan_accruals_df['CR_ACC'].apply(lambda x: 1 if x == "268110000" else 0)

# Select relevant fields
comparison_data = merged_loan_accruals_df[['Account_Number', 'Amount_Tag', 'LCY_Amount', 'INT_SUSPENSE_FLAG']]

# Filter for Unsuspended Interest
unsuspended_interest_data = comparison_data[comparison_data['INT_SUSPENSE_FLAG'] == 0]

# Summarize relevant fields
grouped_data = unsuspended_interest_data.groupby(['Account_Number', 'Amount_Tag']).agg({'LCY_Amount': 'sum'}).reset_index()

# Filter for Main Interest Accruals that are not in suspense
main_interest_data = comparison_data[(comparison_data['Amount_Tag'] == 'Main Interest Liquidation') & (comparison_data['INT_SUSPENSE_FLAG'] == 0)]

# Filter for Penal Interest Accruals that are not in suspense
penal_interest_data = comparison_data[(comparison_data['Amount_Tag'] == 'Penal Interest Liquidation') & (comparison_data['INT_SUSPENSE_FLAG'] == 0)]

# Group main_interest_data and penal_interest_data to compute totals
main_interest_total = main_interest_data.groupby(['Account_Number', 'Amount_Tag'])['LCY_Amount'].sum().reset_index()
penal_interest_total = penal_interest_data.groupby(['Account_Number', 'Amount_Tag'])['LCY_Amount'].sum().reset_index()

# Create 'system_computed_interest_df' with both Main Interest and Penal Interest totals
system_computed_interest_df = pd.concat([main_interest_total, penal_interest_total])

# Rename the columns if needed
system_computed_interest_df = system_computed_interest_df.rename(columns={'LCY_Amount': 'System_Computed_Interest'})

system_computed_penal_interest_df = penal_interest_total.copy()

# Rename the 'Amount_Tag' column to 'System_Computed_Penal_Interest'
system_computed_penal_interest_df = system_computed_penal_interest_df.rename(columns={'Amount_Tag': 'System_Computed_Penal_Interest'})

# Now, 'system_computed_interest_df' contains the computed interest totals for 'Main Interest' and 'Penal Interest'.

# Now, you have the necessary dataframes for main interest, penal interest, and grouped data for Step 12.



# Step 12: Computing Variance and Reporting
# Assuming you have computed the interest amounts in Step 7 and stored them in a DataFrame named 'computed_interest_df'.
# Assuming you have the following DataFrames available:
# system_computed_interest_df: System-computed summarized interest per customer
# main_interest_data: DataFrame for Main Interest Accruals (from Step 11)
# penal_interest_data: DataFrame for Penal Interest Accruals (from Step 11)
# grouped_data: Grouped data for unsuspended interest (from Step 11)

# Calculate the summation of effective interest and penal interest for Step 12
grouped_data['Effective_Interest'] = grouped_data['LCY_Amount']
grouped_data['Effective_Penal_Interest'] = grouped_data['LCY_Amount']

# Perform a multiple join on ACCOUNT_NUMBER between the system computed interest and grouped_data
merged_interest_data = pd.merge(grouped_data, system_computed_interest_df, on='Account_Number', how='left')
merged_interest_data = pd.merge(merged_interest_data, system_computed_penal_interest_df, on='Account_Number', how='left')


# Calculate the difference between system-computed and PwC-computed interest for the year
merged_interest_data['Interest_Variance'] = merged_interest_data['System_Computed_Interest'] - merged_interest_data['Effective_Interest']

# Calculate the difference between system-computed and PwC-computed penal interest for the year
merged_interest_data['Penal_Interest_Variance'] = merged_interest_data['System_Computed_Penal_Interest'] - merged_interest_data['Effective_Penal_Interest']

# Calculate the total interest variance for the year
merged_interest_data['Total_Interest_Variance'] = merged_interest_data['System_Computed_Penal_Interest'] + merged_interest_data['System_Computed_Interest'] - merged_interest_data['Effective_Interest'] - merged_interest_data['Effective_Penal_Interest']

# Select relevant fields for reporting
#reporting_data = merged_interest_data[['Account_Number', 'Customer_Name', 'Effective_Interest', 'Effective_Penal_Interest', 'System_Computed_Penal_Interest', 'System_Computed_Interest', 'Interest_Variance', 'Penal_Interest_Variance', 'Total_Interest_Variance']]


# Visualizations


# Assuming you have the interest variances in your 'merged_interest_data' DataFrame
# which includes 'Interest_Variance', 'Penal_Interest_Variance', and 'Total_Interest_Variance' columns

fig = go.Figure()

# Add box plots for each interest variance
fig.add_trace(go.Box(y=merged_interest_data['Interest_Variance'], name='Main Interest Variance'))
fig.add_trace(go.Box(y=merged_interest_data['Penal_Interest_Variance'], name='Penal Interest Variance'))
fig.add_trace(go.Box(y=merged_interest_data['Total_Interest_Variance'], name='Total Interest Variance'))

fig.update_layout(title='Interest Variances Box Plot', yaxis_title='Variance Amount')
st.plotly_chart(fig)


fig = go.Figure()

# Add box plots for each variance
fig.add_trace(go.Box(y=merged_interest_data['Effective_Interest'], name='Effective Interest'))
fig.add_trace(go.Box(y=merged_interest_data['System_Computed_Interest'], name='System-Computed Interest'))
fig.add_trace(go.Box(y=merged_loan_accruals_df['PENAL_INTEREST'], name='Penal Interest'))
fig.add_trace(go.Box(y=merged_interest_data['System_Computed_Penal_Interest'], name='System-Computed Penal Interest'))

fig.update_layout(title='Interest Variances Box Plot', yaxis_title='Interest Amount')
st.plotly_chart(fig)

# Visualization 10: Effective vs. System-Computed Interest (Scatter Plot)
fig10 = go.Figure()
fig10.add_trace(go.Scatter(x=merged_interest_data['Effective_Interest'], y=merged_interest_data['System_Computed_Interest'], mode='markers', text=merged_interest_data['Account_Number'], name='Main Interest'))
fig10.add_trace(go.Scatter(x=merged_interest_data['Effective_Penal_Interest'], y=merged_interest_data['System_Computed_Penal_Interest'], mode='markers', text=merged_interest_data['Account_Number'], name='Penal Interest'))
fig10.update_layout(title='Effective vs. System-Computed Interest')
fig10.update_xaxes(title='Effective Interest and Penal Interest')
fig10.update_yaxes(title='System-Computed Interest and Penal Interest')

# Visualization 9: Loan Portfolio Concentration (Bar Chart)
loan_concentration = loan_listing_df['Product_Description'].value_counts().reset_index()
loan_concentration.columns = ['Product', 'Count']
fig9 = px.bar(loan_concentration, x='Product', y='Count', title='Loan Portfolio Concentration by Product')


# Loan Portfolio Size (Line Chart):
# Assuming you have a DataFrame 'loan_listing_df' with 'Value_Date' and 'Amount_Financed' columns
fig_loan_portfolio_size = px.area(
    loan_listing_df,
    x='Value_Date',
    y='Amount_Financed',
    title='Loan Portfolio Size Over Time',
    labels={'Amount_Financed': 'Loan Portfolio Size'},
    line_shape='spline',  # Use spline interpolation for smoother curves
)

# Show the plot
st.plotly_chart(fig_loan_portfolio_size)

# Visualization 3: Profitability by Loan Product (Bar Chart)
product_profitability_df = loan_listing_df.groupby('Product_Description')['Interest_Overdue_Amount'].mean().reset_index()
fig_product_profitability = px.bar(product_profitability_df, x='Product_Description', y='Interest_Overdue_Amount', title='Profitability by Loan Product')
st.plotly_chart(fig_product_profitability)

# Visualization 4: Early Warning Indicators (Line Chart)
early_warning_df = loan_listing_df.groupby('Value_Date')['Overdue_Days'].mean().reset_index()
fig_early_warning = px.line(early_warning_df, x='Value_Date', y='Overdue_Days', title='Early Warning Indicators Over Time')
st.plotly_chart(fig_early_warning)

# Visualization 5: Loan Growth Rate (Line Chart)
loan_growth_df = loan_listing_df.groupby('Value_Date')['Amount_Financed'].sum().reset_index()
loan_growth_df['Loan_Growth_Rate'] = loan_growth_df['Amount_Financed'].pct_change()
fig_loan_growth = px.line(loan_growth_df, x='Value_Date', y='Loan_Growth_Rate', title='Loan Growth Rate Over Time')
st.plotly_chart(fig_loan_growth)

# Visualization 6: Loan Origination Volume (Bar Chart)
origination_df = loan_listing_df.groupby('Value_Date')['Amount_Financed'].sum().reset_index()
fig_origination_volume = px.bar(origination_df, x='Value_Date', y='Amount_Financed', title='Loan Origination Volume Over Time')
st.plotly_chart(fig_origination_volume)

# Visualization 7: Non-Performing Loan (NPL) Ratio (Line Chart)
npl_df = merged_loan_accruals_df.groupby('Value_Date')['INT_SUSPENSION_FLAG'].mean().reset_index()
fig_npl_ratio = px.line(npl_df, x='Value_Date', y='INT_SUSPENSION_FLAG', title='Non-Performing Loan (NPL) Ratio Over Time')
st.plotly_chart(fig_npl_ratio)

# Visualization 2: Loan Performance by Branch/Region (Bar Chart)
branch_performance_df = loan_listing_df.groupby('Branch_Code')['Interest_Overdue_Amount'].mean().reset_index()
fig_branch_performance = px.bar(branch_performance_df, x='Branch_Code', y='Interest_Overdue_Amount', title='Loan Performance by Branch')
st.plotly_chart(fig_branch_performance)
# Visualization 8: Visualize Repayment Performance Trends by Branch Code
fig_repayment_by_branch = px.line(loan_listing_df, x='Value_Date', y='Overdue_Days', color='Branch_Code',
                                  title='Repayment Performance Trends by Branch Code')
st.plotly_chart(fig_repayment_by_branch)

# Visualization 9: Visualize Repayment Performance Trends by Product Description
fig_repayment_by_product = px.line(loan_listing_df, x='Value_Date', y='Overdue_Days', color='Product_Description',
                                   title='Repayment Performance Trends by Product Description')
st.plotly_chart(fig_repayment_by_product)

# Visualization 10: Generate a Table for Top and Worst Performers
loan_listing_df_sorted = loan_listing_df.sort_values(by='Overdue_Days')
top_20_performers = loan_listing_df_sorted.head(20)
worst_20_performers = loan_listing_df_sorted.tail(20)

st.write("Top 20 Performers:")
st.write(top_20_performers)

st.write("Worst 20 Performers:")
st.write(worst_20_performers)

# Visualization 11: Interest NOT in Suspense per Branch and Product Description
interest_suspense_df = merged_loan_accruals_df[merged_loan_accruals_df['INT_SUSPENSE_FLAG'] == 0]
suspense_per_branch_product = interest_suspense_df.groupby(['Branch_Code', 'Product_Description']).size().reset_index(name='Count')
fig_suspense_per_branch_product = px.bar(suspense_per_branch_product, x='Branch_Code', y='Count', color='Product_Description',
                                         title='Interest NOT in Suspense per Branch and Product Description',
                                         labels={'Count': 'Number of Accounts'})
st.plotly_chart(fig_suspense_per_branch_product)


overdue_df = loan_listing_df[loan_listing_df['Overdue_Days'] > 0]

# Group by 'Branch_Code', 'Product_Description', and calculate the count of overdue days
overdue_per_branch_product = overdue_df.groupby(['Branch_Code', 'Product_Description'])['Overdue_Days'].count().reset_index()

# Create a stacked bar plot
fig_overdue_per_branch_product = px.bar(
    overdue_per_branch_product,
    x='Branch_Code',
    y='Overdue_Days',
    color='Product_Description',
    title='Overdue Days per Branch and Product Description',
    labels={'Overdue_Days': 'Number of Overdue Days'}
)

# Show the plot
st.plotly_chart(fig_overdue_per_branch_product)

# Visualization 2: Loan Maturity Distribution (Histogram)
fig2 = px.histogram(loan_listing_df, x='Maturity_Date', title='Loan Maturity Distribution')

# Visualization 3: Loan Repayment Patterns (Line Chart)
merged_loan_accruals_df['Schedule_Due_Date'] = pd.to_datetime(merged_loan_accruals_df['Schedule_Due_Date'])
repayment_patterns = merged_loan_accruals_df.groupby('Schedule_Due_Date')['LCY_Amount'].sum().reset_index()
fig3 = px.line(repayment_patterns, x='Schedule_Due_Date', y='LCY_Amount', title='Loan Repayment Patterns Over Time')

# Visualization 4: Loan Aging Analysis (Bar Chart)
loan_listing_df['Overdue_Days'] = loan_listing_df['Overdue_Days'].fillna(0)
overdue_counts = loan_listing_df['Overdue_Days'].value_counts().sort_index()
fig4 = px.bar(overdue_counts, x=overdue_counts.index, y=overdue_counts.values, title='Loan Aging Analysis')
