import math
import numpy as np
from scipy.stats import shapiro, norm
from scipy import stats
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from scipy import stats
import numpy as np
def calculate_sample_size(alpha, beta, allocation_ratio, effect_size):
    # Standardized effect size
    standardized_effect_size = effect_size / math.sqrt(1 / (2 * allocation_ratio))
def calculate_sample_size(alpha, beta, allocation_ratio, effect_size):
    # Standardized effect size
    standardized_effect_size = effect_size / math.sqrt(1 / (2 * allocation_ratio))

    # Calculate sample size
    n = (norm.ppf(1 - alpha / 2) + norm.ppf(1 - beta)) ** 2 / (standardized_effect_size ** 2)

    # Adjust for unequal allocation ratio
    n *= 2 * allocation_ratio / (1 + allocation_ratio)

    return math.ceil(n)
def main():
    alpha = float(input("Enter alpha (e.g., 0.05): "))
    beta = float(input("Enter beta (e.g., 0.2): "))
    allocation_ratio = float(input("Enter allocation ratio (e.g., 1 for equal groups): "))
    effect_size = float(input("Enter effect size (e.g., 0.3 for small effect): "))

    sample_size = calculate_sample_size(alpha, beta, allocation_ratio, effect_size)

    print(f"Recommended sample size: {sample_size}")

if __name__ == "__main__":
    main()
 def check_parametric_assumptions(data, column):
    numerical_data = pd.to_numeric(data[column], errors='coerce').dropna()
    shapiro_stat, shapiro_p = stats.shapiro(numerical_data)
    return shapiro_p > 0.05
def check_data_transformation(data, column):
    numerical_data = pd.to_numeric(data[column], errors='coerce').dropna()
    log_data = np.log(numerical_data)
    shapiro_stat, shapiro_p = stats.shapiro(log_data)
    return shapiro_p > 0.05
def perform_statistical_test(data, column, test_type, num_groups, paired):
    if test_type == "F-max test, Brown and Smythe's test, Bartlett's test":
        groups = [data[data['Group'] == group][column] for group in data['Group'].unique()]
        f_stat, p_value = stats.bartlett(*groups)
        print("Bartlett's test statistic:", f_stat)
        print("p-value:", p_value)
    elif test_type == "Chi-Square Test (One/Two Sample)":
        contingency_table = pd.crosstab(data[column], data['Group'])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print("Chi-Square statistic:", chi2_stat)
        print("p-value:", p_value)
        print("Degrees of freedom:", dof)

    elif test_type == "Student's unpaired t-test":
        groups = [data[data['Group'] == group][column] for group in data['Group'].unique()]
        t_stat, p_value = stats.ttest_ind(*groups)
        print("t-statistic:", t_stat)
        print("p-value:", p_value)
    elif test_type == "Paired t-test":
        groups = [data[data['Group'] == group][column] for group in data['Group'].unique()]
        t_stat, p_value = stats.ttest_rel(*groups)
        print("t-statistic:", t_stat)
        print("p-value:", p_value)
    elif test_type == "Parametric ANOVA":
        groups = [data[data['Group'] == group][column] for group in data['Group'].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        print("F-statistic:", f_stat)
        print("p-value:", p_value)
    elif test_type == "Non-parametric Kruskal-Wallis Test":
        groups = [data[data['Group'] == group][column] for group in data['Group'].unique()]
        h_stat, p_value = stats.kruskal(*groups)
        print("H-statistic:", h_stat)
        print("p-value:", p_value)
    elif test_type == "Mann-Whitney U or Wilcoxon Rank Sums test":
        group1 = data[data['Group'] == data['Group'].unique()[0]][column]
        group2 = data[data['Group'] == data['Group'].unique()[1]][column]
        u_stat, p_value = stats.mannwhitneyu(group1, group2)
        print("U-statistic:", u_stat)
        print("p-value:", p_value)
    else:
        print("Error: Unsupported test type.")


def recommend_statistical_test():
    print("Welcome to the Statistical Test Recommender!")
    print("\n1. What kind of data do you have?")
    print("A) Discrete (e.g., counts, categories)")
    print("B) Continuous (e.g., measurements, scores)")
    data_type = input("Enter your choice (A/B): ")
    if data_type == "A":
        return "Chi-Square Test (One/Two Sample)"
    print("\n2. What type of question do you have?")
    print("A) Analyze relationships")
    print("B) Compare groups")
    question_type = input("Enter your choice (A/B): ")
    if question_type == "A":
        print("\n3. Do you have a true independent variable?")
        print("A) Yes")
        print("B) No")
        independent_variable = input("Enter your choice (A/B): ")
        if independent_variable == "A":
            return "Regression Analysis"
        else:
            print("\n4. Please enter the path to your CSV file:")
            file_path = input()
            try:
                data = pd.read_csv(file_path)
                columns = list(data.columns)
                print("\nSelect the column to analyze:")
                for i, column in enumerate(columns):
                    print(f"{i+1}. {column}")
                column_choice = int(input("Enter the column number: "))
                column = columns[column_choice - 1]
                parametric_assumptions_satisfied = check_parametric_assumptions(data, column)
                if parametric_assumptions_satisfied:
                    return "Pearson's r"
                else:
                    data_transform_worked = check_data_transformation(data, column)
                    if data_transform_worked:
                        return "Pearson's r"
                    else:
                        return "Spearman's Rank Correlation"
            except FileNotFoundError:
                return "Error: File not found. Please check the file path and try again."
    elif question_type == "B":
        print("\n3. What is your question?")
        print("A)Testing for equal variances")
        print("B)Comparing mean")
        question = input("Enter your choice (A/B): ")
        if question == "A":
            return "F-max test, Brown and Smythe's test, Bartlett's test"
        elif question == "B":
            print("\n4. How many groups are there?")
            print("A) 2 groups")
            print("B) More than two groups")
            num_groups = input("Enter your choice (A/B): ")
            print("\n5. Please enter the path to your CSV file:")
            file_path = input()
            try:
                data = pd.read_csv(file_path)
                columns = list(data.columns)
                print("\nSelect the column to analyze:")
                for i, column in enumerate(columns):
                    print(f"{i+1}. {column}")
                column_choice = int(input("Enter the column number: "))
                column = columns[column_choice - 1]

                # Perform statistical test
                if num_groups == "A":
                     print("\n6. Is this a paired test?")
                     print("A) Yes")
                     print("B) No")
                     paired = input("Enter your choice (A/B): ")
                     if paired == "A":
                          return "Paired t-test"
                     else:
                          return "Student's unpaired t-test"
                elif num_groups == "B":
                    return "Parametric ANOVA, followed by Tukey's or Bonferroni's post-hoc test (if significant)"
            except FileNotFoundError:
              return "Error: File not found. Please check the file path and try again."
            else:
              return "Error: Invalid choice. Please check your input."

def main():
    recommendation = recommend_statistical_test()
    print("\nRecommended statistical test:", recommendation)

    file_path = input("Please enter the path to your CSV file: ")
    try:
        data = pd.read_csv(file_path)
        columns = list(data.columns)
        print("\nSelect the column to analyze:")
        for i, column in enumerate(columns):
            print(f"{i+1}. {column}")
        column_choice = int(input("Enter the column number: "))
        column = columns[column_choice - 1]
        num_groups = input("How many groups are there? (Enter a number): ")
        perform_statistical_test(data, column, recommendation, int(num_groups), False)

    except FileNotFoundError:
          print("Error: File not found. Please check the file path and try again.")

if __name__ == "__main__":
    main()
def statistical_analysis(file_path):
    # Read CSV file
    data = pd.read_csv(file_path)

    # Select column to analyze
    column_names = list(data.columns)
    print("Select a column to analyze:")
    for i, name in enumerate(column_names):
        print(f"{i+1}. {name}")
    choice = int(input("Enter the number of your chosen column: "))
    column = column_names[choice - 1]

    # Statistical analysis
    values = data[column]
    mean = np.mean(values)
    median = np.median(values)

    # Handle mode calculation for unique values
    mode_result = stats.mode(values)

    # Check if mode_result is a scalar value
    if isinstance(mode_result, np.ndarray):
        mode = mode_result[0]
    else:
        mode = mode_result

    std_dev = np.std(values)
    variance = np.var(values)
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)

    # P-value calculation (Shapiro-Wilk test for normality)
    p_value = stats.shapiro(values)[1]

    # Print results
    print("\nStatistical Analysis Results:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")
    print(f"P-value (Shapiro-Wilk test): {p_value}")

   

file_path = input("Enter the path to your CSV file: ")
statistical_analysis(file_path)
import pandas as pd
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Define the parameters
n_control = 5
n_intervention = 5
mean_control = 25
mean_intervention = 30
std_dev = 2

# Generate the data
control_data = np.random.normal(mean_control, std_dev, n_control)
intervention_data = np.random.normal(mean_intervention, std_dev, n_intervention)

# Create a DataFrame
data = pd.DataFrame({
    'ID': range(1, n_control + n_intervention + 1),
    'Group': ['Control'] * n_control + ['Intervention'] * n_intervention,
    'Pre_Intervention': np.concatenate([np.random.normal(mean_control, std_dev, n_control), np.random.normal(mean_intervention, std_dev, n_intervention)]),
    'Post_Intervention': np.concatenate([control_data, intervention_data])
})

# Save the DataFrame to a CSV file
data.to_csv('interventional_rct_data.csv', index=False)




