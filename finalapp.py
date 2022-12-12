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
st.markdown("Input your demographics below to find out!")

user_age = st.slider("How old are you?", min_value = 10, max_value = 97, value = 20, step = 1)
st.write("I am", user_age, "years old.")

user_gender = st.selectbox("Gender?", options = [1,0])
st.caption("If female, choose 1. Otherwise, 0")

user_married = st.selectbox("Are you married?", options = [1,0])
st.caption("If yes, choose 1. Otherwise, 0.")

user_parent = st.selectbox("Are you a parent?", options = [1,0])
st.caption("If yes, choose 1. Otherwise, 0.")

user_education= st.slider("What is your highest education level?", min_value = 1, max_value = 8, value = 3, step = 1)
st.caption("1 = Less than high school | 2 = High school incomplete | 3 = High school graduate | 4 = Some college, no degree | 5 = Two-year associate degree from a college or university | 6 = Four-year college or university degree/Bachelors degree  | 7 = Some postgraduate or professional schooling, no postgraduate degree | 8 = Postgraduate or professional degree, including masters, doctorate, medical or law degree")

user_income_option = st.selectbox("What is your income range?", options = [0,1,2,3,4,5,6,7,8,9])
st.caption("1 = Less than \$10,000 | 2 = 10 to under \$20,000 | 3 = 20 to under \$30,000 | 4 = 30 to under \$40,000 | 5 = 40 to under \$50,000 | 6 = 50 to under \$75,000 | 7 = 75 to under \$100,000  | 8 = 100 to under \$150,000 | 9 = \$150,000 or more")

#user input array
person_input = [user_income_option, user_education, user_parent, user_married, user_gender, user_age]

predicted_class = lr.predict([person_input])
probs_user = lr.predict_proba([person_input])

st.subheader("Are you a LinkedIn User?")

if predicted_class == 1:
    label="You are a user! :confetti_ball:"
else:
    label = "You are not a user... :frowning:"
st.write(label)

st.write("Given your inputs, the probability you were a user ", probs_user)
st.caption("0 = probability you were not a user | 1 = probability you were a user")