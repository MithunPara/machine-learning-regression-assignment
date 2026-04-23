Happiness & Abalone Regression Models

This repository contains a machine learning course assignment focused on implementing regression models.

Overview

This project implements regression models from scratch using Python and basic libraries.
	•	Part 1: Linear regression using gradient descent compared with Ordinary Least Squares (OLS) on a Happiness vs GDP dataset.
	•	Part 2: Linear and polynomial regression models to predict abalone age (rings) based on physical features.


SETUP

Commands
conda create -n ml-regression python=3.10
conda activate ml-regression

- creates conda environment with Python version 

conda install pandas matplotlib numpy seaborn

- install dependencies into the environment, the read/write operations are UTF-8

RUNNING FILES
Part 1

python3 part1_happiness_regression.py
What it does:
	•	Trains a linear regression model using gradient descent
	•	Compares results with OLS
	•	Outputs:
	•	Multiple regression lines (different learning rates & epochs)
	•	Comparison plot (Gradient Descent vs OLS)
	•	Learned β (beta) values

Part 2

python3 part2_abalone_regression.py
What it does:
	•	Visualizes relationships between features and target (rings)
	•	Applies linear or polynomial regression based on observed trends
	•	Splits data into training/testing sets
	•	Outputs:
	•	Feature vs target plots
	•	Model fit on training and testing data
	•	Learned β (beta) values

Notes
	•	Ensure dataset paths in the scripts are correctly set before running
	•	The training/testing split size in Part 2 can be adjusted within the script
	•	All file operations use UTF-8 encoding


REFERENCES

	•	Gradient Descent (lecture concepts)
	•	Polynomial regression concepts
	•	External resources:
	•	GeeksForGeeks (gradient descent, numpy usage)
	•	StackOverflow (matplotlib plotting)
	•	Various Python documentation sources