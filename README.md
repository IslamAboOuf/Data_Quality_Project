# Data Quality App

This is a Python-based web application built using
**Streamlit** for performing common data quality tasks such as handling missing values, duplicates,
and outliers in datasets, The app also integrates with **Ollama** for a chatbot
interface to interact with the dataset and answer questions using a **Retrieval-Augmented Generation (RAG)** model.

## Features

### 1. Data Quality Tasks:
- Handle missing values by filling, dropping, or flagging them.
- Identify and remove duplicate records.
- Detect and treat outliers using statistical methods
 
### 2. Chatbot Integration:
- Leverages the **Ollama** platform for interacting with datasets.
- Uses **RAG** to provide intelligent answers and insights based on the dataset.

### 3. User-Friendly Interface:
- Built with **Streamlit**, providing an intuitive and interactive web UI.


- Upload your dataset (CSV or Excel) via the sidebar.
- Select the task you want to perform from the navigation menu in the sidebar:
  - **Dataset Info**: View basic information about your dataset (columns, types, non-null counts).
  - **Describe Dataset**: View the descriptive statistics of the dataset.
  - **Handle Missing Values**: Choose to fill or drop missing values from columns.
  - **Handle Duplicates**: Identify and remove duplicate rows.
  - **Handle Outliers**: Remove outliers using the IQR method.
  - **Chat using RAG**: Interact with your dataset via a chatbot powered by Ollama.

After performing any changes, you can download the modified dataset.

## Demo Video
For a detailed walkthrough of the project, watch the [demo video here](https://youtu.be/SFPLEq2ZPHo).

## Technologies Used
- **Streamlit:** For building the web interface.
- **Pandas & NumPy:** For data manipulation and analysis.
- **Ollama API:** For integrating the chatbot and RAG model.
- **Python:** The programming language for the app.




