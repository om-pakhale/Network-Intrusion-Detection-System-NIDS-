import streamlit as st
import pandas as pd
st.header("Financial Fraud & CDR Analyzer")
st.info("Upload Bank Logs, UPI Transactions, or Call Detail Records (CDR) for instant pattern analysis.")

upload_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"])

if upload_file is not None:
            st.write("Analyzing records...")
            try:
                if upload_file.name.endswith('.csv'):
                    df = pd.read_csv(upload_file)
                else:
                    df = pd.read_excel(upload_file)

                st.success(f"Successfully loaded {len(df)} records into the database.")
    
                with st.expander("View Raw Data Preview"):
                    st.dataframe(df.head(10))
                
                st.subheader("1. Identify Repeat Suspects")
                id_col = st.selectbox("Select the column containing Phone Numbers / UPI IDs:", df.columns)
                
                if st.button("Scan for Repeat Offenders"):
                    repeats = df[id_col].value_counts().head(10)
                    st.error("🚨 Top 10 Most Frequent Entities Detected:")
                    st.dataframe(repeats)


                st.subheader("2. Filter High-Value Transactions")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    amt_col = st.selectbox("Select the column containing Transaction Amounts (₹):", numeric_cols)
                    threshold = st.number_input("Enter Minimum Amount Threshold (₹):", min_value=0, value=50000)
                    
                    if st.button("Filter High-Value Targets"):
                        high_value_df = df[df[amt_col] >= threshold]
                        st.warning(f"Found {len(high_value_df)} transactions above ₹{threshold}!")
                        st.dataframe(high_value_df)
                else:
                    st.write("No numeric columns detected for amount filtering.")


                st.subheader("3. Instant Suspect Cross-Reference")
                search_query = st.text_input("Enter specific UPI ID, Bank Account, or Phone Number to search:")
                
                if search_query:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
                    results = df[mask]
                    
                    if not results.empty:
                        st.success(f"Target Acquired: Found {len(results)} matching records!")
                        st.dataframe(results)
                    else:
                        st.success("Target Clear: No matches found in this dataset.")

            except Exception as e:
                st.error(f"Error processing the file: Please ensure it is a valid CSV or Excel file. Details: {e}")