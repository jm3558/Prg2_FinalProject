import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



#Q1
s = pd.read_csv("social_media_usage.csv")

#Q2
def clean_sm(x):
    y = np.where(x == 1, 1, 0)
    return (y)

#Q3
ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]), 
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]), 
    "parent":np.where(s["par"] == 1, 1, 0), 
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0), 
    "age":np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()

#Q4
# Y as target vector and X as feature set
y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female","age"]]

#Q5

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                   stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=989) # set for reproducibility

#Q6

#Initialize algorithm 
lr = LogisticRegression(class_weight = "balanced")
lr.fit(x_train, y_train)

#Q7
y_pred = lr.predict(x_test)

#Q8
confusion_matrix(y_test, y_pred)

#Q9 classification report (precision, recall, f1score)
print(classification_report(y_test, y_pred))

#Q10 Streamlit predictions

st.header("Are you a LinkedIn user?")
st.subheader("Input your demographics below to find out!")

user_age = st.slider("How old are you?", min_value = 10, max_value = 97, value = 20, step = 1)
st.write("I am", user_age, "years old.")

gender_input = st.selectbox("What is your gender?", options = ["Female", "Male"])

if gender_input == "Female":
    gender_input = 1
else: gender_input = 0

married_input = 0

if st.checkbox("Are you *married*? Check this box!"):
    married_input = 1

parent_input = 0

if st.checkbox("Are you *a parent*? Check this box!"):
    parent_input = 1

educ_input = st.selectbox("Education Level", options = [
    "Less than high school", 
    "High school incomplete", 
    "High school graduate", 
    "Some college, no degree", 
    "Two-year associate degree from a college or university", 
    "Some postgraduate or professional schooling, no postgraduate degree",
    "Four-year college or university degree/Bachelors degree", 
    "Postgraduate or professional degree, including masters, doctorate, medical or law degree"])

if educ_input == "Less than high school":
    educ_input = 1
elif educ_input == "High school incomplete":
    educ_input = 2
elif educ_input == "High school graduate":
    educ_input = 3
elif educ_input == "Some college, no degree":
    educ_input = 4
elif educ_input == "Two-year associate degree from a college or university":
    educ_input = 5
elif educ_input == "Some postgraduate or professional schooling, no postgraduate degree":
    educ_input = 6
elif educ_input == "Four-year college or university degree/Bachelors degree":
    educ_input = 7
elif educ_input == "Postgraduate or professional degree, including masters, doctorate, medical or law degree":
    educ_input = 8

income_input = st.selectbox("Income Range", options = [
    "Less than $10,000", 
    "10 to under $20,000", 
    "20 to under $30,000", 
    "30 to under 40,000", 
    "40 to under $50,000", 
    "50 to under $75,000",
    "75 to under $100,000", 
    "100 to under $150,000"])

if income_input == "Less than $10,000":
    income_input = 1
elif income_input == "10 to under $20,000":
    income_input = 2
elif income_input == "20 to under $30,000":
    income_input = 3
elif income_input == "30 to under $40,000":
    income_input = 4
elif income_input == "40 to under $50,000":
    income_input = 5
elif income_input == "50 to under $75,000":
    income_input = 6
elif income_input == "75 to under $100,000":
    income_input = 7
elif income_input == "100 to under $150,000":
    income_input = 8

#user input array
person_input = [income_input, educ_input, parent_input, married_input, gender_input, user_age]

predicted_class = lr.predict([person_input])
probs_user = lr.predict_proba([person_input])

st.subheader("So are you a LinkedIn User...?")

if predicted_class == 1:
    label="Congrats, you are a user! :confetti_ball:"
else:
    label = "Sorry, you are not a user... :frowning:"
st.markdown(label)

st.write("Given your inputs, the probability you are a LinkedIn user is", round(round(probs_user[0][1],3)*100, 2), "%")
st.write("The likelihood you were not a LinkedIn user is", round(round(probs_user[0][0],3)*100, 2), "%")