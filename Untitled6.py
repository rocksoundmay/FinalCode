#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Check Working Directory
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

current_directory = os.getcwd()
print(current_directory)


# In[8]:


# Change Working Directory

new_path = "/Users/Kenzie/Downloads"
os.chdir(new_path)
updated_dir = os.getcwd()
print(updated_dir)


# In[31]:


# Import Data

def read_patient_data(file_path):
    try:
        # Read data from the file and skip the header
        with open(file_path, 'r') as file:
            next(file)  # Skip the header
            lines = file.readlines()

        # Parse each line and extract relevant information
        data = []
        for line in lines:
            values = line.strip().split(',')
            patient_id = int(values[0])
            readmission = int(values[1])
            satisfaction_scores = [int(val) for val in values[2:]]
            row = [patient_id, readmission] + satisfaction_scores
            data.append(row)

        return np.array(data)

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


# In[32]:


# Function to Calculate Statistics

def calculate_statistics(data):
    num_readmitted = np.sum(data[:, 1])
    avg_satisfaction_scores = np.mean(data[:, 2:], axis=0)
    return num_readmitted, avg_satisfaction_scores


# In[33]:


# Logistic Regression

def perform_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# In[37]:


# Plotting the Regression Curve

def plot_logistic_regression_curve(X, y, model, hospital_name):
    plt.figure(figsize=(8, 6))

    # Plot the data points
    plt.scatter(X, y, color='blue', label='Actual Data')

    # Plot the logistic regression curve
    x_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_prob = model.predict_proba(x_values)[:, 1]
    plt.plot(x_values, y_prob, color='red', label='Logistic Regression Curve')

    plt.title(f'Logistic Regression Curve - {hospital_name}')
    plt.xlabel('Overall Satisfaction Scores')
    plt.ylabel('Probability of Readmission')
    plt.legend()
    plt.show()


# In[39]:


# Main Function

def main():
    file_path = 'Hospital1.txt' # Hospital One
    
    try:
        # Read patient data
        patient_data = read_patient_data('Hospital1.txt')
        
        # Calculate statistics
        num_readmitted, avg_satisfaction_scores = calculate_statistics(patient_data)
        
        # Display statistics
        print(f"Number of Patients Readmitted: {num_readmitted}")
        print(f"Average Staff Satisfaction: {avg_satisfaction_scores[0]:.2f}")
        print(f"Average Cleanliness Satisfaction: {avg_satisfaction_scores[1]:.2f}")
        print(f"Average Food Satisfaction: {avg_satisfaction_scores[2]:.2f}")
        print(f"Average Comfort Satisfaction: {avg_satisfaction_scores[3]:.2f}")
        print(f"Average Communication Satisfaction: {avg_satisfaction_scores[4]:.2f}\n")
        
        # Prepare data for logistic regression
        X = patient_data[:, 2:].mean(axis=1).reshape(-1, 1)  # Overall satisfaction scores
        y = patient_data[:, 1]  # Readmission
        
        # Perform logistic regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display confusion matrix and classification report
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot logistic regression curve
        plot_logistic_regression_curve(X, y, model, 'Hospital One')
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        file_path = 'Hospital1.txt' # Hospital Two
    
    try:
        # Read patient data
        patient_data = read_patient_data('Hospital2.txt')
        
        # Calculate statistics
        num_readmitted, avg_satisfaction_scores = calculate_statistics(patient_data)
        
        # Display statistics
        print(f"Number of Patients Readmitted: {num_readmitted}")
        print(f"Average Staff Satisfaction: {avg_satisfaction_scores[0]:.2f}")
        print(f"Average Cleanliness Satisfaction: {avg_satisfaction_scores[1]:.2f}")
        print(f"Average Food Satisfaction: {avg_satisfaction_scores[2]:.2f}")
        print(f"Average Comfort Satisfaction: {avg_satisfaction_scores[3]:.2f}")
        print(f"Average Communication Satisfaction: {avg_satisfaction_scores[4]:.2f}\n")
        
        # Prepare data for logistic regression
        X = patient_data[:, 2:].mean(axis=1).reshape(-1, 1)  # Overall satisfaction scores
        y = patient_data[:, 1]  # Readmission
        
        # Perform logistic regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display confusion matrix and classification report
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot logistic regression curve
        plot_logistic_regression_curve(X, y, model, 'Hospital Two')
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


# In[42]:


# Comparing Hospital One and Hospital Two

def compare_hospitals(file_path_one, file_path_two):
    try:
        # Read patient data for Hospital One
        patient_data_one = read_patient_data(file_path_one)

        # Calculate statistics for Hospital One
        num_readmitted_one, avg_satisfaction_scores_one = calculate_statistics(patient_data_one)

        # Display statistics for Hospital One
        print(f"Number of Patients Readmitted (Hospital One): {num_readmitted_one}")
        print(f"Average Staff Satisfaction (Hospital One): {avg_satisfaction_scores_one[0]:.2f}")
        print(f"Average Cleanliness Satisfaction (Hospital One): {avg_satisfaction_scores_one[1]:.2f}")
        print(f"Average Food Satisfaction (Hospital One): {avg_satisfaction_scores_one[2]:.2f}")
        print(f"Average Comfort Satisfaction (Hospital One): {avg_satisfaction_scores_one[3]:.2f}")
        print(f"Average Communication Satisfaction (Hospital One): {avg_satisfaction_scores_one[4]:.2f}\n")

        # Prepare data for logistic regression for Hospital One
        X_one = patient_data_one[:, 2:].mean(axis=1).reshape(-1, 1)  # Overall satisfaction scores
        y_one = patient_data_one[:, 1]  # Readmission

        # Perform logistic regression for Hospital One
        X_train_one, X_test_one, y_train_one, y_test_one = train_test_split(X_one, y_one, test_size=0.2, random_state=42)
        model_one = LogisticRegression()
        model_one.fit(X_train_one, y_train_one)

        # Read patient data for Hospital Two
        patient_data_two = read_patient_data(file_path_two)

        # Calculate statistics for Hospital Two
        num_readmitted_two, avg_satisfaction_scores_two = calculate_statistics(patient_data_two)

        # Display statistics for Hospital Two
        print(f"Number of Patients Readmitted (Hospital Two): {num_readmitted_two}")
        print(f"Average Staff Satisfaction (Hospital Two): {avg_satisfaction_scores_two[0]:.2f}")
        print(f"Average Cleanliness Satisfaction (Hospital Two): {avg_satisfaction_scores_two[1]:.2f}")
        print(f"Average Food Satisfaction (Hospital Two): {avg_satisfaction_scores_two[2]:.2f}")
        print(f"Average Comfort Satisfaction (Hospital Two): {avg_satisfaction_scores_two[3]:.2f}")
        print(f"Average Communication Satisfaction (Hospital Two): {avg_satisfaction_scores_two[4]:.2f}\n")

        # Prepare data for logistic regression for Hospital Two
        X_two = patient_data_two[:, 2:].mean(axis=1).reshape(-1, 1)  # Overall satisfaction scores
        y_two = patient_data_two[:, 1]  # Readmission

        # Perform logistic regression for Hospital Two
        X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(X_two, y_two, test_size=0.2, random_state=42)
        model_two = LogisticRegression()
        model_two.fit(X_train_two, y_train_two)

        # Display confusion matrix and classification report for Hospital One
        print("Confusion Matrix (Hospital One):")
        print(confusion_matrix(y_test_one, model_one.predict(X_test_one)))
        print("\nClassification Report (Hospital One):")
        print(classification_report(y_test_one, model_one.predict(X_test_one)))

        # Display confusion matrix and classification report for Hospital Two
        print("\nConfusion Matrix (Hospital Two):")
        print(confusion_matrix(y_test_two, model_two.predict(X_test_two)))
        print("\nClassification Report (Hospital Two):")
        print(classification_report(y_test_two, model_two.predict(X_test_two)))

        # Plot logistic regression curves for both hospitals
        plot_logistic_regression_curves(X_one, y_one, model_one, X_two, y_two, model_two)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_logistic_regression_curves(X_one, y_one, model_one, X_two, y_two, model_two):
    plt.figure(figsize=(10, 6))

    # Plot the data points for Hospital One
    plt.scatter(X_one, y_one, color='blue', label='Hospital One Data', alpha=0.6)

    # Plot the logistic regression curve for Hospital One
    x_values_one = np.linspace(X_one.min(), X_one.max(), 100).reshape(-1, 1)
    y_prob_one = model_one.predict_proba(x_values_one)[:, 1]
    plt.plot(x_values_one, y_prob_one, color='red', label='Hospital One Regression Curve')

    # Plot the data points for Hospital Two
    plt.scatter(X_two, y_two, color='green', label='Hospital Two Data', alpha=0.6)

    # Plot the logistic regression curve for Hospital Two
    x_values_two = np.linspace(X_two.min(), X_two.max(), 100).reshape(-1, 1)
    y_prob_two = model_two.predict_proba(x_values_two)[:, 1]
    plt.plot(x_values_two, y_prob_two, color='purple', label='Hospital Two Regression Curve')

    plt.title('Logistic Regression Curves Comparison')
    plt.xlabel('Overall Satisfaction Scores')
    plt.ylabel('Probability of Readmission')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare_hospitals('Hospital1.txt', 'Hospital2.txt')


# In[ ]:




