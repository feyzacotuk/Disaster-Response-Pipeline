import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    
    Args:
    messages_filepath (str): Filepath for the csv file containing messages.
    categories_filepath (str): Filepath for the csv file containing categories.
    
    Returns:
    df (DataFrame): Merged DataFrame containing messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged DataFrame.
    
    Args:
    df (DataFrame): Merged DataFrame containing messages and categories data.
    
    Returns:
    df (DataFrame): Cleaned DataFrame with categories split into separate columns and duplicates removed.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    # Drop rows with a value of 2 in any category
    for column in categories:
        categories = categories[categories[column] != 2]
    
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Sanity checks, excluding 'id' column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('id')
    print(f"Categories with values greater than 1: {df[numeric_cols].apply(lambda x: x[x > 1].count())}")
    assert df.duplicated().sum() == 0, "There are still duplicates in the dataset"
    assert (df[numeric_cols] > 1).sum().sum() == 0, "There are values greater than 1 in the dataset"

    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame into a SQLite database.
    
    Args:
    df (DataFrame): Cleaned DataFrame containing messages and categories data.
    database_filename (str): Filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute the ETL pipeline: load data, clean data, and save data to a database.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
