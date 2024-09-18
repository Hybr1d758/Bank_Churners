import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    logging.info("Starting the script")

    # Load your datasets
    logging.info("Loading the dataset")
    data = pd.read_csv("/Users/edwardjr/Documents/Data Science/Google/BankChurners.csv")
    df = pd.DataFrame(data)
    logging.info(f"Dataset loaded. Shape: {df.shape}")

    # Display the first few rows
    logging.info("Displaying the first few rows of the dataset")
    print(df.head())

    # Display dataset info
    logging.info("Displaying dataset info")
    df.info()

    # Check for missing values
    logging.info("Checking for missing values")
    missing_values = df.isnull().sum()
    missing_percentages = 100 * df.isnull().sum() / len(df)
    missing_table = pd.concat([missing_values, missing_percentages], axis=1, keys=['Missing Values', 'Percentage Missing'])
    print("\nMissing Values Summary:")
    print(missing_table)

    # Log the number of columns with missing values
    columns_with_missing = missing_values[missing_values > 0]
    logging.info(f"Number of columns with missing values: {len(columns_with_missing)}")
    
    if len(columns_with_missing) > 0:
        logging.info("Columns with missing values:")
        for column, count in columns_with_missing.items():
            logging.info(f"  {column}: {count} missing values ({missing_percentages[column]:.2f}%)")

        # Fill null values
        logging.info("Filling null values")

        # Separate numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns

        # Fill numeric columns with median
        imputer_numeric = SimpleImputer(strategy='median')
        df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])

        # Fill categorical columns with mode
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])

        logging.info("Null values filled")

        # Check if all null values are filled
        remaining_nulls = df.isnull().sum().sum()
        logging.info(f"Remaining null values after filling: {remaining_nulls}")

    else:
        logging.info("No missing values found in the dataset")

    # Check for duplicates
    logging.info("Checking for duplicate rows")
    duplicates = df.duplicated()
    duplicate_count = duplicates.sum()
    
    logging.info(f"Number of duplicate rows: {duplicate_count}")
    
    if duplicate_count > 0:
        logging.info("Displaying first few duplicate rows:")
        print(df[duplicates].head())
        
        # Check for duplicates based on specific columns
        logging.info("Checking for duplicates based on specific columns (excluding ID columns)")
        columns_to_check = df.columns.drop(['CLIENTNUM'])  # CLIENTNUM is an ID column
        partial_duplicates = df.duplicated(subset=columns_to_check, keep=False)
        partial_duplicate_count = partial_duplicates.sum()
        
        logging.info(f"Number of partial duplicate rows (excluding ID columns): {partial_duplicate_count}")
        
        if partial_duplicate_count > 0:
            logging.info("Displaying first few partial duplicate rows:")
            print(df[partial_duplicates].head())
    else:
        logging.info("No duplicate rows found in the dataset")

    # Group by 'Customer_Age' and 'Gender', and calculate various statistics
    age_gender_stats = df.groupby(['Customer_Age', 'Gender']).agg({
        'Credit_Limit': ['mean', 'median', 'min', 'max'],
        'Total_Trans_Amt': ['mean', 'sum'],
        'Total_Trans_Ct': ['mean', 'sum']
    })

    logging.info("Age and Gender aggregation complete")
    print("\nAge and Gender Statistics:")
    print(age_gender_stats.head())

     # Group by 'Education_Level' and calculate statistics
    education_stats = df.groupby('Education_Level').agg({
        'Credit_Limit': ['mean', 'median'],
        'Total_Trans_Amt': ['mean', 'sum'],
        'Total_Trans_Ct': ['mean', 'sum']
    })

    logging.info("Education Level aggregation complete")
    print("\nEducation Level Statistics:")
    print(education_stats)

    # Group by 'Income_Category' and calculate statistics
    income_stats = df.groupby('Income_Category').agg({
        'Credit_Limit': ['mean', 'median'],
        'Total_Trans_Amt': ['mean', 'sum'],
        'Total_Trans_Ct': ['mean', 'sum']
    })

    logging.info("Income Category aggregation complete")
    print("\nIncome Category Statistics:")
    print(income_stats)

    # Calculate churn rate by different categories
    df['Churn_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
    
    churn_by_category = df.groupby(['Education_Level', 'Income_Category', 'Card_Category'])['Churn_Flag'].mean().reset_index()
    churn_by_category = churn_by_category.sort_values('Churn_Flag', ascending=False)

    logging.info("Churn rate calculation complete")
    print("\nChurn Rate by Category:")
    print(churn_by_category)

    logging.info("Script completed successfully")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
