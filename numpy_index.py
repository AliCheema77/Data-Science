import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Calculate age in years rounded down
# df['age_years'] = (df['age'] // 365).astype(int)

# # Find the age group with the highest average weight
# highest_avg_weight_group = df.groupby('age_years')['weight'].mean().idxmax()
# highest_avg_weight = df[df['age_years'] == highest_avg_weight_group]['weight'].mean()

# # Find the age group with the lowest average weight
# lowest_avg_weight_group = df.groupby('age_years')['weight'].mean().idxmin()
# lowest_avg_weight = df[df['age_years'] == lowest_avg_weight_group]['weight'].mean()

# # Calculate the weight difference in percentage
# weight_difference_percentage = ((highest_avg_weight - lowest_avg_weight) / lowest_avg_weight) * 100

# # Print the result
# print(f"The age group with the highest average weight is {highest_avg_weight_group} years old.")
# print(f"The age group with the lowest average weight is {lowest_avg_weight_group} years old.")
# print(f"The difference in weight is {weight_difference_percentage:.2f}%.")


# df = pd.read_csv("cardio_base.csv")

# # Calculate age in years rounded down
# df['age_years'] = (df['age'] // 365).astype(int)

# # Create a new column indicating whether a person is over 50
# df['over_50'] = df['age_years'] > 50

# # Calculate the average cholesterol levels for people over 50 and under 50
# avg_cholesterol_over_50 = df[df['over_50']]['cholesterol'].mean()
# avg_cholesterol_under_50 = df[~df['over_50']]['cholesterol'].mean()

# # Calculate the percentage difference in cholesterol levels
# percentage_difference = ((avg_cholesterol_over_50 - avg_cholesterol_under_50) / avg_cholesterol_under_50) * 100

# # Print the result
# print(f"The average cholesterol level for people over 50 is {avg_cholesterol_over_50:.2f}.")
# print(f"The average cholesterol level for people under 50 is {avg_cholesterol_under_50:.2f}.")

# # Check if people over 50 have higher cholesterol levels
# if percentage_difference > 0:
#     print(f"Yes, people over 50 have {percentage_difference:.2f}% higher cholesterol levels than the rest.")
# else:
#     print("No, people over 50 do not have higher cholesterol levels than the rest.")



# Are men more likely to be a smoker than women?

# The data contains information to identify gender IDs!
    


# import pandas as pd

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Calculate the smoking rate for men and women
# smoking_rate_men = df[df['gender'] == 2]['smoke'].mean()
# smoking_rate_women = df[df['gender'] == 1]['smoke'].mean()

# # Print the result
# print(f"The smoking rate for men is {smoking_rate_men:.2%}.")
# print(f"The smoking rate for women is {smoking_rate_women:.2%}.")

# # Check if men are more likely to be smokers than women
# if smoking_rate_men > smoking_rate_women:
#     print("Yes, men are more likely to be smokers than women.")
#     times_more = smoking_rate_men / smoking_rate_women
#     print(f"Men are approximately {times_more:.2f} times more likely to be smokers than women.")
# else:
#     print("No, men are not more likely to be smokers than women.")




# import pandas as pd

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Calculate the height of the tallest 1% of people
# tallest_1_percent_height = df['height'].quantile(0.99)

# # Print the result
# print(f"The height of the tallest 1% of people is {tallest_1_percent_height:.2f} cm.")



# "Which two features have the highest spearman rank correlation"



# import pandas as pd

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Calculate the Spearman rank correlation matrix
# spearman_corr = df.corr(method='spearman')

# # Exclude correlation of features with themselves and find the pair with the highest correlation
# highest_corr = (spearman_corr.mask(np.tril(np.ones(spearman_corr.shape)).astype(np.bool))
#                                       .abs()
#                                       .idxmax())

# # Extract the names of the features with the highest correlation
# feature1, feature2 = highest_corr[0], highest_corr[1]

# # Print the result
# print(f"The two features with the highest Spearman rank correlation are '{feature1}' and '{feature2}'.")


# import pandas as pd
# from scipy.stats import spearmanr

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Calculate Spearman rank correlation matrix for all features
# correlation_matrix = df.corr(method='spearman')

# # Find the two features with the highest Spearman rank correlation
# max_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()
# feature1, feature2 = max_corr.index[1]

# # Print the result
# print(f"The two features with the highest Spearman rank correlation are: {feature1} and {feature2}")


# import pandas as pd
# from scipy.stats import spearmanr

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Drop non-numeric columns
# numeric_df = df.select_dtypes(include='number')

# # Remove rows with missing values
# numeric_df = numeric_df.dropna()

# # Calculate Spearman rank correlation matrix for numeric features
# correlation_matrix = numeric_df.corr(method='spearman')

# # Find the two features with the highest Spearman rank correlation
# max_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()
# feature1, feature2 = max_corr.index[1]

# # Print the result
# print(f"The two features with the highest Spearman rank correlation are: {feature1} and {feature2}")


# "What percentage of people are more than two standard deviations far from the average height?"

# import pandas as pd

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Calculate the average height and standard deviation
# average_height = df['height'].mean()
# std_dev_height = df['height'].std()

# # Identify people whose height is more than two standard deviations away from the average
# outliers = df[(df['height'] > average_height + 2 * std_dev_height) | (df['height'] < average_height - 2 * std_dev_height)]

# # Calculate the percentage of outliers
# percentage_outliers = (len(outliers) / len(df)) * 100

# # Print the result
# print(f"The percentage of people more than two standard deviations away from the average height is {percentage_outliers:.2f}%.")


# "What percentage of population over 50 years old consume alcohol?   also use the cardio_alco.csv and merge the dataset on ID, ignore those persons, where we have no alcohol consumption information."


# import pandas as pd

# # Load the datasets with ';' as the delimiter
# df_base = pd.read_csv("cardio_base.csv")
# df_alco = pd.read_csv("cardio_alco.csv", delimiter=';')

# # Standardize column names by removing leading/trailing whitespaces
# df_base.columns = df_base.columns.str.strip()
# df_alco.columns = df_alco.columns.str.strip()

# # Merge the datasets on 'id'
# merged_df = pd.merge(df_base, df_alco, on='id', how='inner')

# # Calculate age in years rounded down
# merged_df['age_years'] = (merged_df['age'] // 365).astype(int)

# # Filter the population over 50 years old
# population_over_50 = merged_df[merged_df['age_years'] > 50]

# # Calculate the percentage of people over 50 consuming alcohol
# percentage_consumers_over_50 = (population_over_50['alco'].sum() / len(population_over_50)) * 100

# # Print the result
# print(f"The percentage of the population over 50 years old consuming alcohol is {percentage_consumers_over_50:.2f}%.")



# give me answer on same dataset "Which of the following statements is true with 95% confidence?"
# 1)Somkers weight less than non smokers
# 2)Somkers have higher cholesterol level than non smokers
# 3)Smoker have higher blood pressure than no smokers
# 4)Men have higher blood pressure than woman


# import pandas as pd
# from scipy.stats import ttest_ind

# # Load the dataset
# df = pd.read_csv("cardio_base.csv")

# # Assuming 'smoker', 'cholesterol', 'bp_high', 'gender', and 'weight' columns exist
# # Perform t-tests for the given statements

# # 1) Smokers weigh less than non-smokers
# weight_smokers = df[df['smoke'] == 1]['weight']
# weight_non_smokers = df[df['smoke'] == 0]['weight']
# t_stat_weight, p_value_weight = ttest_ind(weight_smokers, weight_non_smokers)

# # 2) Smokers have higher cholesterol levels than non-smokers
# cholesterol_smokers = df[df['smoke'] == 1]['cholesterol']
# cholesterol_non_smokers = df[df['smoke'] == 0]['cholesterol']
# t_stat_cholesterol, p_value_cholesterol = ttest_ind(cholesterol_smokers, cholesterol_non_smokers)

# # 3) Smokers have higher blood pressure than non-smokers
# bp_smokers = df[df['smoke'] == 1]['ap_hi']
# bp_non_smokers = df[df['smoke'] == 0]['ap_hi']
# t_stat_bp, p_value_bp = ttest_ind(bp_smokers, bp_non_smokers)

# # 4) Men have higher blood pressure than women
# bp_men = df[df['gender'] == 2]['ap_hi']
# bp_women = df[df['gender'] == 1]['ap_hi']
# t_stat_bp_gender, p_value_bp_gender = ttest_ind(bp_men, bp_women)

# # Print the results
# print(f"1) Smokers weigh less than non-smokers: {p_value_weight < 0.05}")
# print(f"2) Smokers have higher cholesterol levels than non-smokers: {p_value_cholesterol < 0.05}")
# print(f"3) Smokers have higher blood pressure than non-smokers: {p_value_bp < 0.05}")
# print(f"4) Men have higher blood pressure than women: {p_value_bp_gender < 0.05}")

# true_statements = [i+1 for i, p_value in enumerate([p_value_weight, p_value_cholesterol, p_value_bp, p_value_bp_gender]) if p_value < 0.05]

# print("\nThe true statement(s) with 95% confidence:", true_statements)



# Second Dataset, Covid19 cases

# This dataset contains daily covid19 cases for all countries in the world. Each row represents a calendar day. The rows also contain some simple information about the countries, like population, percentage of the population over 65, GDP and hospital beds per thousand inhabitants. Please use this dataset to answer the following questions.

# When did the difference in the total number of confirmed cases between Italy and Germany become more than 10 000?


# import pandas as pd

# # Load the dataset
# covid_data = pd.read_csv("covid_data.csv")

# # Filter data for Italy and Germany
# italy_data = covid_data[covid_data['location'] == 'Italy']
# germany_data = covid_data[covid_data['location'] == 'Germany']

# # Merge the datasets on 'date' to compare the total confirmed cases
# merged_data = pd.merge(italy_data, germany_data, on='date', suffixes=('_italy', '_germany'))

# # Calculate the difference in total confirmed cases
# merged_data['confirmed_cases_difference'] = merged_data['new_cases_italy'] - merged_data['new_cases_germany']

# # Find the date when the difference becomes more than 10,000
# date_difference_10k = merged_data.loc[merged_data['confirmed_cases_difference'] < 10000, 'date'].min()

# # Print the result
# print(f"The difference in total confirmed cases between Italy and Germany became more than 10,000 on {date_difference_10k}.")




# Look at the cumulative number of confirmed cases in Italy between 2020-02-28 and 2020-03-20. Fit an exponential function (y = Ae^(Bx)) to this set to express cumulative cases as a function of days passed, by minimizing squared loss. 

# What is the difference between the exponential curve and the total number of real cases on 2020-03-20


# import pandas as pd
# import numpy as np
# from scipy.optimize import curve_fit

# # Load the dataset
# covid_data = pd.read_csv("covid_data.csv")

# # Filter data for Italy and the specified date range
# italy_data = covid_data[(covid_data['location'] == 'Italy') & (covid_data['date'] >= '2020-02-28') & (covid_data['date'] <= '2020-03-20')]

# # Extract the relevant columns
# days_passed = np.arange(1, len(italy_data) + 1)  # Days passed since 2020-02-28
# cumulative_cases = italy_data['new_cases'].cumsum()  # Cumulative new cases

# if not cumulative_cases.empty:
#     # Define the exponential function
#     def exponential_function(x, A, B):
#         return A * np.exp(B * x)

#     # Fit the exponential function to the data
#     params, covariance = curve_fit(exponential_function, days_passed, cumulative_cases)

#     # Extract the fitted parameters
#     A, B = params

#     # Calculate the predicted cumulative cases using the exponential curve
#     predicted_cases = exponential_function(days_passed, A, B)

#     # Calculate the difference between the predicted and real cases on 2020-03-20
#     date_2020_03_20_index = len(cumulative_cases) - 1
#     real_cases_2020_03_20 = cumulative_cases.iloc[date_2020_03_20_index]
#     predicted_cases_2020_03_20 = exponential_function(date_2020_03_20_index + 1, A, B)
#     difference = predicted_cases_2020_03_20 - real_cases_2020_03_20

#     print(f"The difference between the exponential curve and the real cases on 2020-03-20 is approximately {difference:.2f} cases.")
# else:
#     print("Insufficient data for Italy within the specified date range.")


# import pandas as pd

# # Load the dataset
# covid_data = pd.read_csv("covid_data.csv")

# # Group the data by country and calculate the total deaths and population
# grouped_data = covid_data.groupby('location').agg({
#     'new_deaths': 'max',
#     'population': 'max'
# })

# # Calculate death rate per million inhabitants
# grouped_data['death_rate_per_million'] = (grouped_data['new_deaths'] / grouped_data['population']) * 1_000_000

# # Sort the data by death rate in descending order
# sorted_data = grouped_data.sort_values(by='death_rate_per_million', ascending=False)

# # Get the name of the country with the 3rd highest death rate
# third_highest_country = sorted_data.index[2]

# print(f"The country with the 3rd highest death rate is: {third_highest_country}")



# What is the F1 score of the following statement:

# Countries, where more than 20% of the population is over 65 years old, have death rates over 50 per million inhabitants. 

# Ignore countries, where any of the necessary information is missing!




# import pandas as pd
# from sklearn.metrics import f1_score

# # Load the dataset
# covid_data = pd.read_csv("covid_data.csv")

# # Filter data to include only necessary information
# filtered_data = covid_data.dropna(subset=['population', 'aged_65_older_percent', 'new_deaths'])

# # Calculate death rate per million inhabitants
# filtered_data['death_rate_per_million'] = (filtered_data['new_deaths'] / filtered_data['population']) * 1_000_000

# # Define the condition based on the statement
# condition = (filtered_data['aged_65_older_percent'] > 20) & (filtered_data['death_rate_per_million'] > 50)

# # Create a binary label indicating whether the condition is met
# labels_true = condition.astype(int)

# # Simulate a prediction based on the statement (for demonstration purposes, replace with a more accurate prediction)
# # In a real-world scenario, you might use a predictive model
# # For simplicity, let's assume all countries are predicted as meeting the condition
# labels_pred = pd.Series(1, index=filtered_data.index)

# # Calculate the F1 score
# f1 = f1_score(labels_true, labels_pred)

# print(f"The F1 score for the given statement is: {f1:.2f}")


import pandas as pd

# Load the dataset
covid_data = pd.read_csv("covid_data.csv")

# Filter data for countries with at least 5 hospital beds per 1000 inhabitants
filtered_data = covid_data[covid_data['hospital_beds_per_thousand'] >= 5]

# Calculate the total number of countries with at least 5 hospital beds per 1000 inhabitants
total_countries_with_5_beds = len(filtered_data)

# Calculate the number of countries with GDP over $10,000 and at least 5 hospital beds per 1000 inhabitants
countries_with_gdp_over_10000 = filtered_data[filtered_data['gdp_per_capita'] > 10000]
countries_with_gdp_over_10000_and_5_beds = len(countries_with_gdp_over_10000)

# Calculate the conditional probability
probability_gdp_over_10000_given_5_beds = countries_with_gdp_over_10000_and_5_beds / total_countries_with_5_beds

print(f"The probability that a country has GDP over $10,000 given at least 5 hospital beds per 1000 inhabitants is: {probability_gdp_over_10000_given_5_beds:.2%}")
