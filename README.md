# AI-Powered Breast Cancer Prediction System 🏥

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F8991D?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-279434?style=for-the-badge)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An advanced, multi-page web application that leverages a pre-trained Scikit-learn model for breast cancer risk assessment and integrates Google's Gemini Pro model to provide users with actionable, AI-generated recommendations.
Streamlit : https://breast-cancer-predictor-bd9ztwavwmztvkz5zurjkq.streamlit.app/
This project serves as a comprehensive demonstration of building end-to-end machine learning applications, from data input and model inference to interactive visualization and a generative AI-powered feedback loop.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏛️ Architecture & Technology Stack](#️-architecture--technology-stack)
- [🔬 The Dataset & Model](#-the-dataset--model)
- [🚀 Local Development Setup](#-local-development-setup)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [⚠️ Medical Disclaimer](#️-medical-disclaimer)

---

## 📝 Overview

The AI-Powered Breast Cancer Prediction System is designed to provide an intuitive interface for a complex machine learning task. It allows users—ranging from students to researchers—to interact with a predictive model based on the Wisconsin Breast Cancer dataset. Users can input 30 cellular characteristics and receive an immediate risk assessment.

Beyond a simple prediction, the application enhances user understanding by:
1.  **Visualizing** the risk score and the most influential factors behind the prediction.
2.  **Leveraging a Large Language Model (LLM)** to interpret the results and provide context-aware lifestyle and follow-up recommendations.
3.  **Maintaining a session log** that can be reviewed and exported, facilitating analysis and record-keeping.

The multi-page design ensures a clean, organized workflow, guiding the user from introduction through data entry, analysis, and reporting.

---

## ✨ Key Features

- **Multi-Page Navigation:** A clean sidebar separates the application into logical sections: Introduction, Patient Input, Prediction Visuals, AI Recommendations, and History.
- **Versatile Data Input:** Supports three methods for data entry: manual input for all 30 features, one-click loading of pre-defined samples (High Risk, Low Risk), and batch processing via CSV file upload.
- **Real-Time Model Inference:** Utilizes a pre-trained Logistic Regression model to deliver instantaneous predictions and probability scores upon user request.
- **Interactive Data Visualization:** Powered by Plotly, the app features:
    - A **Risk Gauge** for an immediate, intuitive understanding of the malignancy probability.
    - A **Feature Importance Chart** that dynamically displays the top factors contributing to the current prediction.
    - A **Session Summary Pie Chart** showing the distribution of predictions.
- **Generative AI Recommendations:** Integrates with the Google Gemini model via LangChain to translate the numerical prediction into easy-to-understand, personalized advice.
- **Session Logging and Export:** All predictions are timestamped and logged in a history table, which can be cleared or exported to a CSV file for external analysis.

---

## 🏛️ Architecture & Technology Stack

This application is built with a modern Python stack, integrating machine learning, generative AI, and web development frameworks.

- **Frontend & UI:**
  - **[Streamlit](https://streamlit.io/):** The core framework used to build the interactive, multi-page web application with pure Python.

- **Machine Learning & Data Handling:**
  - **[Scikit-learn](https://scikit-learn.org/):** Used for loading and running the pre-trained Logistic Regression model.
  - **[Pandas](https://pandas.pydata.org/):** For robust data manipulation, primarily for creating DataFrames from user input to feed into the model.
  - **[NumPy](https://numpy.org/):** For efficient numerical operations.
  - **[Joblib](https://joblib.readthedocs.io/):** For serializing and deserializing the Scikit-learn model file (`.joblib`).

- **Generative AI Integration:**
  - **[Google Generative AI](https://ai.google.com/):** Provides the Gemini model for generating human-like recommendations.
  - **[LangChain](https://www.langchain.com/):** Acts as the orchestration framework to connect to the Gemini API, manage prompts, and parse the output seamlessly.

- **Data Visualization:**
  - **[Plotly](https://plotly.com/python/):** Used to create rich, interactive, and aesthetically pleasing charts (gauge, bar, pie) that are natively supported by Streamlit.

---

## 🔬 The Dataset & Model

### The Dataset

The model was trained on the **[Wisconsin Breast Cancer (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)** from the UCI Machine Learning Repository.

- **Content:** It contains 569 instances with 30 numeric features computed from a digitized image of a fine-needle aspirate (FNA) of a breast mass.
- **Features:** These describe characteristics of the cell nuclei present in the image, such as radius, texture, perimeter, area, smoothness, and concavity.
- **Objective:** The goal is to classify a tumor as either **Malignant (M)** or **Benign (B)** based on these features.

### The Machine Learning Model

The application uses a pre-trained **Logistic Regression** model.
- **File:** `logistic_regression_model.joblib`
- **Rationale:** Logistic Regression is a robust and interpretable algorithm for binary classification tasks, making it an excellent choice for this problem. It provides not only a classification (Malignant/Benign) but also a probability score, which is crucial for the risk assessment feature.

---

## 🚀 Local Development Setup

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

- [Python](https://www.python.org/downloads/) (version 3.9 or higher)
- [Git](https://git-scm.com/downloads/) for cloning the repository

### 2. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
