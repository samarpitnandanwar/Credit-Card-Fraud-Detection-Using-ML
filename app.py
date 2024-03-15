import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)


# UI part -->>
# create Streamlit app


if __name__ == "__main__":
    st.set_page_config(page_title="Option Menu", layout="wide")
    selected = option_menu(None, ["Home", "Summary", 'About'], 
        icons=['house', 'file-earmark-text', 'people'], 
        menu_icon="cast", default_index=0, orientation="horizontal")

    if selected == "Home":
        st.title("Credit Card Fraud Detection Model")
        st.subheader("Enter the following features to check if the transaction is legitimate or fraudulent:")

        # create input fields for user to enter feature values
        input_df = st.text_input('Input All features')
        input_df_lst = input_df.split(',')
        # create a button to submit input and get prediction
        submit = st.button("Submit")

        if submit:
            # get input feature values
            features = np.array(input_df_lst, dtype=np.float64)
            # make prediction
            prediction = model.predict(features.reshape(1,-1))
            # display result
            if prediction[0] == 0:
                st.success('Legitimate transaction', icon="✅")
            else:
                st.error('Fraudulent transaction', icon="🚨")

        # Custom HTML and CSS for the footer
        footer_style = f"""
            <style>
                .footer {{
                    background: #ff4b4b;
                    color: white;
                    padding: 10px;
                    position: fixed;
                    bottom: 0;
                    width: 90vw;
                    text-align: center;
                    font-size: 20px;
                }}
            </style>
        """

        # Display the custom HTML
        st.markdown(footer_style, unsafe_allow_html=True)

        # Your Streamlit app content goes here

        # Display the footer
        def img_to_bytes(img_path):
            img_bytes = Path(img_path).read_bytes()
            encoded = base64.b64encode(img_bytes).decode()
            return encoded
        def img_to_html(img_path):
            img_html = "<img src='data:image/png;base64,{}' style='width: 100px; padding-left: 20px; padding-right: 20px' class='img-fluid'>".format(
            img_to_bytes(img_path)
            )
            return img_html

        st.markdown(f"<div class='footer'>{img_to_html('assets/SGBAU.png')} Design and Developed By : Final Year Students, Computer Science and Engineering, H.V.P.M. Amravati. {img_to_html('assets/HVPM.png')}<div>", unsafe_allow_html=True)


        
        
    if selected == "Summary":
        st.title("Credit Card Fraud Detection Abstract")
        # Function to display the PDF of a given file
        def displayPDF(file):
            try:
                # Check if the file is a URL or a local file path
                if urlparse(file).scheme:
                    # If it has a scheme (http, https, etc.), treat it as a URL
                    with urllib.request.urlopen(file) as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                else:
                    # If it doesn't have a scheme, treat it as a local file path
                    with open(file, 'rb') as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                # Embedding PDF in HTML
                pdf_display = F'<div style="text-align: center; width: 90vw;"><iframe src="data:application/pdf;base64,{base64_pdf}" width="1000" height="1250" type="application/pdf"></iframe></div>'

                # Displaying File
                st.markdown(pdf_display, unsafe_allow_html=True)

            except urllib.error.URLError as e:
                st.error(f"Error: Unable to fetch the PDF file. {e}")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

        # Example usage for a local file path
        file_path = "Synopsis.pdf"
        displayPDF(file_path)

        
    if selected == "About":
        st.title(f"Group Members")

        st.header("Samarpeet Nandanwar") 
        st.subheader("Roll No: 48, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Adib Khan") 
        st.subheader("Roll No: 07, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Suraj Pawar") 
        st.subheader("Roll No: 57, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Deven Malekar") 
        st.subheader("Roll No: 14, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Himanshu kadu") 
        st.subheader("Roll No: 22, Final Year,  Computer Science & Engineering")