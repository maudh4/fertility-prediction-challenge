"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the predict_outcomes function. 


The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

The script can be run from the command line using the following command:

python script.py input_path 

An example for the provided test is:

python script.py data/test_data_liss_2_subjects.csv
"""

import os
import sys
import argparse
import pandas as pd
from joblib import load

parser = argparse.ArgumentParser(description="Process and score data.")
subparsers = parser.add_subparsers(dest="command")

# Process subcommand
process_parser = subparsers.add_parser("predict", help="Process input data for prediction.")
process_parser.add_argument("input_path", help="Path to input data CSV file.")
process_parser.add_argument("--output", help="Path to prediction output CSV file.")

# Score subcommand
score_parser = subparsers.add_parser("score", help="Score (evaluate) predictions.")
score_parser.add_argument("prediction_path", help="Path to predicted outcome CSV file.")
score_parser.add_argument("ground_truth_path", help="Path to ground truth outcome CSV file.")
score_parser.add_argument("--output", help="Path to evaluation score output CSV file.")

args = parser.parse_args()


def predict_outcomes(df):
    """Process the input data and write the predictions."""

    # The predict_outcomes function accepts a Pandas DataFrame as an argument
    # and returns a new DataFrame with two columns: nomem_encr and
    # prediction. The nomem_encr column in the new DataFrame replicates the
    # corresponding column from the input DataFrame. The prediction
    # column contains predictions for each corresponding nomem_encr. Each
    # prediction is represented as a binary value: '0' indicates that the
    # individual did not have a child during 2020-2022, while '1' implies that
    # they did.

    #Column selection
    import pandas as pd

    #Split data
    from sklearn.model_selection import train_test_split

    #Column selection
    from sklearn.compose import make_column_selector as selector
    from sklearn.compose import ColumnTransformer

    #Feature selection
    from sklearn.feature_selection import SelectKBest, f_classif
    #Model
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier 
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import GradientBoostingClassifier

    keepcols = ['nomem_encr',
            'gebjaar', 
            'geslacht',
            'herkomstgroep2017', 
            'herkomstgroep2018', 
            'herkomstgroep2019', 
            'oplmet2017', 
            'oplmet2018', 
            'oplmet2019', 
            'aantalki2017', 
            'aantalki2018', 
            'aantalki2019', 
            'positie2017',
            'positie2018', 
            'positie2019', 
            'leeftijd2017',
            'leeftijd2018', 
            'leeftijd2019', 
            'aantalhh2017', 
            'aantalhh2018', 
            'aantalhh2019', 
            'lftdhhh2017',
            'lftdhhh2018', 
            'lftdhhh2019',
            'partner2017',
            'partner2018',
            'partner2019',
            'belbezig2017', 
            'belbezig2018',
            'belbezig2019',
            'brutohh_f2017',
            'brutohh_f2018', 
            'brutohh_f2019',
            #'cf17j454', 
            'cf18k454',
            #'cf19l454', 
            #'cf17j455', 
            'cf18k455',
            #'cf19l455',
            #'cf17j130', 
            'cf18k130',
            #'cf19l130',
            #'cf17j129',
            'cf18k129',
            #'cf19l129',
            'cf17j128',
            'cf18k128',
            'cf19l128', 
            #'cf17j407', 
            #'cf18k407',
            #'cf19l407', 
            #'cf17j408', 
            #'cf18k408',
            #'cf19l408',
           ]
    df = df.loc[:, keepcols]


    # #### Categorical versus Numerical features
    #Categorical variables 
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(df)

    #Numerical variables
    numerical_columns_selector = selector(dtype_exclude=object)
    numerical_columns = numerical_columns_selector(df)

    for col in numerical_columns:
        col_mean = df[col].mean()
        df[col] = df[col].fillna(col_mean)

    for col in categorical_columns:
       df[col] = df[col].fillna('none')
    
    # Load your trained model from the models directory
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "GBS_feature_selection.joblib")
    model = load(model_path)

    # Use your trained model for prediction
    predictions = model.predict(df)
    # Return the result as a Pandas DataFrame with the columns "nomem_encr" and "prediction"
    return pd.concat([df['nomem_encr'], pd.Series(predictions, name="prediction")], axis=1)


def predict(input_path, output):
    if output is None:
        output = sys.stdout
    df = pd.read_csv(input_path, encoding="latin-1", encoding_errors="replace", low_memory=False)
    predictions = predict_outcomes(df)
    assert (
        predictions.shape[1] == 2
    ), "Predictions must have two columns: nomem_encr and prediction"
    # Check for the columns, order does not matter
    assert set(predictions.columns) == set(
        ["nomem_encr", "prediction"]
    ), "Predictions must have two columns: nomem_encr and prediction"

    predictions.to_csv(output, index=False)


def score(prediction_path, ground_truth_path, output):
    """Score (evaluate) the predictions and write the metrics.
    
    This function takes the path to a CSV file containing predicted outcomes and the
    path to a CSV file containing the ground truth outcomes. It calculates the overall 
    prediction accuracy, and precision, recall, and F1 score for having a child 
    and writes these scores to a new output CSV file.

    This function should not be modified.
    """

    if output is None:
        output = sys.stdout
    # Load predictions and ground truth into dataframes
    predictions_df = pd.read_csv(prediction_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # Merge predictions and ground truth on the 'id' column
    merged_df = pd.merge(predictions_df, ground_truth_df, on="nomem_encr", how="right")

    # Calculate accuracy
    accuracy = len(
        merged_df[merged_df["prediction"] == merged_df["new_child"]]
    ) / len(merged_df)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 1)]
    )
    false_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 0)]
    )
    false_negatives = len(
        merged_df[(merged_df["prediction"] == 0) & (merged_df["new_child"] == 1)]
    )

    # Calculate precision, recall, and F1 score
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    # Write metric output to a new CSV file
    metrics_df = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score]
    })
    metrics_df.to_csv(output, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.input_path, args.output)
    elif args.command == "score":
        score(args.prediction_path, args.ground_truth_path, args.output)
    else:
        parser.print_help()
        predict(args.input_path, args.output)  
        sys.exit(1)
