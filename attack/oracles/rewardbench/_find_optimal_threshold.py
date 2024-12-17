import pandas as pd
import numpy as np

# Load the CSV file
file_path = './oracles/results/IMP_oracle_eval.csv'
data = pd.read_csv(file_path)
data['diff'] = (data['original_score_A'] - data['original_score_B']).abs() 

# Define a function to calculate accuracy for a given threshold
def calculate_accuracy_for_threshold(threshold, data):
    # Predict tie if the absolute difference between scores is less than or equal to the threshold
    data['predicted_label'] = np.where(data['diff'].abs() <= threshold, 'ResponseQuality.TIE', 
                                       np.where(data['original_score_A'] > data['original_score_B'], 
                                                'ResponseQuality.A_BETTER', 
                                                'ResponseQuality.B_BETTER'))
    
    # Calculate the accuracy based on the predicted labels
    data['correct'] = np.where(data['predicted_label'] == data['original_label'], 1, 0)
    accuracy = data['correct'].mean()
    return accuracy

oracles = ["OffsetBiasOracle", "InternLMOracle", "ArmoRMOracle"]

for oracle in oracles:

    df = data[data["oracle_class"] == oracle]

    # Find the optimal threshold by iterating over a range of possible thresholds
    thresholds = np.linspace(0, df['diff'].max(), 100)
    accuracies = [calculate_accuracy_for_threshold(thresh, df.copy()) for thresh in thresholds]

    # Identify the optimal threshold
    optimal_threshold = thresholds[np.argmax(accuracies)]
    optimal_accuracy = max(accuracies)

    # Output the results
    print(f"Oracle: {oracle}")
    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"Maximum Accuracy: {optimal_accuracy}")
