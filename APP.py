import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ollama

# Upload File
def upload_file(): 
    st.title("Data Cleaning and Visualization App")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv") 
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("File uploaded successfully!")

# Display Data Info
def display_data_info():
    st.subheader("Data Information")
    df = st.session_state.df
    st.write("**Columns and Data Types:**")
    st.write(df.dtypes)

    st.write("**Data Discription:**")
    st.write(st.session_state.df.describe())

# Handle Missing Values
def handle_missing_values():
    if "df" not in st.session_state:
        df = st.session_state.df

    df_session = st.session_state.df.copy()  

    st.subheader("Handle Columns")

    st.write("#### Missing Values:-")
    st.write(df_session.isnull().sum())

    columns_with_missing = [col for col in df_session.columns if df_session[col].isnull().sum() > 0]
    if not columns_with_missing:
        st.success("No missing values found!")

    st.write("#### Preview of Data Before Changes:")
    st.dataframe(df_session)

    col = st.selectbox("Select column to handle:", df_session.columns)

    action = st.selectbox(
        f"Select action for column '{col}':",
        [
            "No Action", 
            "Convert to Numeric", 
            "Fill with Mean", 
            "Fill with Median", 
            "Fill with Mode", 
            "Drop Column", 
            "Show Unique Values",
        ],
        key=f"action_{col}"
    )

    
    if action != "No Action":
            if action == "Convert to Numeric":
                df_session[col] = pd.to_numeric(
                    df_session[col].replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                )
                st.write(f"Converted column '{col}' to numeric. Invalid values replaced with NaN.")
            elif action in ["Fill with Mean", "Fill with Median"]:
            # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(df_session[col]):
                    if action == "Fill with Mean":
                        df_session[col].fillna(df_session[col].mean(), inplace=True)
                    elif action == "Fill with Median":
                        df_session[col].fillna(df_session[col].median(), inplace=True)
                    st.write(f"Preview: Applied '{action}' to column '{col}'.")
                else:
                        st.error(f"Cannot apply '{action}' to column '{col}' because it is not numeric.")
            elif action == "Fill with Mode":
            # Mode works on both numeric and object columns
                mode_value = df_session[col].mode()
                if not mode_value.empty:
                    df_session[col].fillna(mode_value[0], inplace=True)
                    st.write(f"Preview: Applied '{action}' to column '{col}'.")
                else:
                    st.error(f"Cannot apply '{action}' to column '{col}' because it has no mode.")
            elif action == "Drop Column":
                df_session.drop(columns=[col], inplace=True)
                st.write(f"Column '{col}' has been dropped.")
            elif action == "Show Unique Values":
                unique_values = df_session[col].unique()
                num_unique_values = df_session[col].nunique()
                st.write(f"Number of Unique Values in '{col}': {num_unique_values}")
                st.write(unique_values)

            # Display the preview DataFrame
    st.write("#### Preview of Changes:")
    st.dataframe(df_session)

    # Confirmation button to apply changes
    if st.button("Confirm and Apply Changes"):
        st.session_state.df = df_session.copy()
        st.success(f"Changes to column '{col}' have been confirmed!")
        st.write(st.session_state.df.head())  # Display updated DataFrame

# Rename Columns
def rename_columns():
    st.subheader("Rename Columns")
    df = st.session_state.df

    columns = df.columns.tolist()
    column_to_rename = st.selectbox("Select Column to Rename", columns)
    new_name = st.text_input("Enter New Name")

    if st.button("Rename"):
        df.rename(columns={column_to_rename: new_name}, inplace=True)
        st.success(f"Column '{column_to_rename}' renamed to '{new_name}'!")
        st.session_state.df = df
        st.write(df.head())

def handle_duplicates():
    
    
    df_session = st.session_state.df  # Reference the session-state DataFrame

    st.subheader("Handle Duplicates")
    
    # Check and show number of duplicates
    duplicates = df_session.duplicated().sum()
    
    # If duplicates exist, provide option to remove them
    if duplicates > 0:
        st.write(f"Number of duplicate rows: {duplicates}")

        if st.button("Remove Duplicates"):
            # Remove duplicates and save the changes back to session state
            df_session.drop_duplicates(inplace=True)
            st.success("Duplicates removed!")
            st.write(f"Number of duplicate rows after removal: {df_session.duplicated().sum()}")
            

            # Save the modified DataFrame back to session state
            st.session_state.df = df_session
    else:
        st.write(f"No duplicate rows found in the dataset.")


# Heatmap
def show_heatmap():
    st.subheader("Heatmap of Correlation")
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_cols.empty:
        corr_matrix = numeric_cols.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns to show correlation.")

# Handle Outliers
def handle_outliers():
    df_session = st.session_state.df

    st.subheader("Handle Outliers")

    outlier_columns = []
    for column in df_session.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df_session[column].quantile(0.25)
        Q3 = df_session[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_session[(df_session[column] < lower_bound) | (df_session[column] > upper_bound)]
        if not outliers.empty:
            outlier_columns.append(column)

    if outlier_columns:
        st.write(f"Columns with outliers: {', '.join(outlier_columns)}")
    else:
        st.success("No outliers detected in any numeric columns.")

    selected_column = st.selectbox("Select a column to handle:", outlier_columns)

    if selected_column:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x=df_session[selected_column], ax=ax)
        ax.set_title(f"Boxplot of {selected_column} (Before Handling Outliers)")
        st.pyplot(fig)

        if st.button(f"Remove Outliers in '{selected_column}'"):
            Q1 = df_session[selected_column].quantile(0.25)
            Q3 = df_session[selected_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_session[selected_column] = df_session[selected_column].where(
                (df_session[selected_column] >= lower_bound) & (df_session[selected_column] <= upper_bound),
                df_session[selected_column].median()
            )

            st.success(f"Outliers removed in '{selected_column}'!")

            st.write(f"**{selected_column}** - Descriptive Statistics After Removing Outliers:")
            st.write(df_session[selected_column].describe())

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.boxplot(x=df_session[selected_column], ax=ax)
            ax.set_title(f"Boxplot of {selected_column} (After Handling Outliers)")
            st.pyplot(fig)

            st.session_state.df = df_session


# Visualization
def Show_Visualization():
    df = st.session_state.df
    column = st.selectbox("Select Column for Visualization", df.columns)

    if df[column].dtype in ['int64', 'float64']:

        # ECDF (Empirical Cumulative Distribution Function)
        st.write(f"### ECDF of {column}")
        plt.figure(figsize=(8, 4))
        sns.ecdfplot(df[column])
        plt.xlabel(column)
        plt.ylabel("ECDF")
        st.pyplot(plt)

        # Histogram
        st.write(f"### Histogram of {column}")
        plt.figure(figsize=(8, 4))
        plt.hist(df[column], bins=30, color="skyblue")
        plt.xlabel(column)
        plt.ylabel("Count")
        st.pyplot(plt)

        # Boxplot
        st.write(f"### Boxplot of {column}")
        plt.figure(figsize=(8, 4))
        sns.boxplot(y=df[column])
        st.pyplot(plt)

        # Density Plot for Selected Column
        st.write(f"### Density Plot of {column}")
        plt.figure(figsize=(8, 4))
        sns.kdeplot(df[column], shade=True, color="green")
        plt.xlabel(column)
        st.pyplot(plt)

        # Line Plot (if applicable)
        st.write(f"### Line Plot of {column}")
        plt.figure(figsize=(8, 4))
        plt.plot(df[column], color="purple")
        plt.xlabel("Index")
        plt.ylabel(column)
        st.pyplot(plt)

        # Scatter Plot (if another numeric column is selected)
        second_column = st.selectbox("Select Second Numeric Column for Scatter Plot", df.columns)
        if df[second_column].dtype in ['int64', 'float64']:
            st.write(f"### Scatter Plot of {column} vs {second_column}")
            plt.figure(figsize=(8, 4))
            plt.scatter(df[column], df[second_column], color="orange")
            plt.xlabel(column)
            plt.ylabel(second_column)
            st.pyplot(plt)

    else:
        # Bar Plot for Categorical Columns
        st.write(f"### Count Plot of {column}")
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[column])
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Histogram
        st.write(f"### Histogram of {column}")
        plt.figure(figsize=(8, 4))
        plt.hist(df[column], bins=30, color="skyblue")
        plt.xlabel(column)
        plt.ylabel("Count")
        st.pyplot(plt)

#RAG
def chat_with_rag():
    st.subheader("Chat using RAG")

    def ollama_generate(query: str, model: str = "llama3.2:latest") -> str:
        """Generate a response using Ollama."""
        try:
            result = ollama.chat(model=model, messages=[{"role": "user", "content": query}])
            return result.get("message", {}).get("content", "No response content.")
        except Exception as e:
            return f"Error: {e}"

    # Function to chat with CSV using Ollama
    def chat_with_csv_ollama(df, prompt, model="llama3.2:latest", max_rows=10):
        """Chat with a CSV using Ollama."""
        # Summarize dataset: Include column names, row count, and sample rows
        summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        column_info = "Columns:\n" + "\n".join([f"- {col} (type: {str(df[col].dtype)})" for col in df.columns])
        sample_data = f"Sample rows:\n{df.head(5).to_string(index=False)}"

        # Include data content (limit rows if necessary)
        data_content = f"The dataset:\n{df.head(max_rows).to_string(index=False)}"

        # Create the query
        query = f"""
        You are a data assistant. Here is the summary of the dataset:
        {summary}
        {column_info}
        {sample_data}

        {data_content}

        Based on this dataset, answer the following question:
        {prompt}
        """
        return ollama_generate(query, model=model)

    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # Stores the history as a list of dictionaries with roles and messages


        
    df = st.session_state.df
    if df is not None:
        
        user_input = st.chat_input("Ask a question:")

        if user_input:
            # Add user query to the conversation
            st.session_state.conversation.append({"role": "user", "content": user_input})

            # Generate response from Ollama
            with st.spinner("Generating response..."):
                assistant_response = chat_with_csv_ollama(df, user_input)

            # Add assistant response to the conversation
            st.session_state.conversation.append({"role": "assistant", "content": assistant_response})

        # Display the conversation
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            elif message["role"] == "assistant":
                # Check if the message contains code blocks
                if "```" in message["content"]:
                    # Split by code blocks
                    code_blocks = message["content"].split("```")
                    for i, block in enumerate(code_blocks):
                        if i % 2 == 1:  # Odd indices are code blocks
                            st.code(block.strip(), language="python")  # Render as code
                        else:
                            if block.strip():  # Avoid rendering empty text
                                st.chat_message("assistant").markdown(block.strip())
                else:
                    st.chat_message("assistant").markdown(message["content"])


# Download Data
def download_data():
    st.subheader("Download Data")
    df = st.session_state.df
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

# Main Function
def main():
    st.sidebar.title("Operations ")
    options = st.sidebar.radio(" ", [
        "Upload File", "Data Info", "Handle Missing Values", "Rename Columns",
        "Handle Outliers","Handle Duplicates", "Heatmap", "Visualization","Chat With AI", "Download Data"
    ])

    if options == "Upload File":
        upload_file()
    elif "df" in st.session_state:
        if options == "Data Info":
            display_data_info()
        elif options == "Handle Missing Values":
            handle_missing_values()
        elif options == "Rename Columns":
            rename_columns()
        elif options == "Handle Outliers":
            handle_outliers()
        elif options == "Handle Duplicates":
            handle_duplicates()
        elif options == "Heatmap":
            show_heatmap()
        elif options == "Visualization":
            Show_Visualization()
        elif options =="Chat With AI":
            chat_with_rag()
        elif options == "Download Data":
            download_data()

    else:
        st.warning("Please upload a file first.")

if __name__ == "__main__":
    main()






