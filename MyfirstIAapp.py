import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

st.title("Flower iris category prediction App")

st.sidebar.header("Inputs for prediction algorithm")

def inputs_params():
    sepal_length=st.sidebar.slider("Sepal length",4.3,8.9,5.0)
    sepal_width=st.sidebar.slider("Sepal width",2.0,4.4,3.0)
    petal_length=st.sidebar.slider("Petal length",1.0,6.9,2.0)
    petal_width =st.sidebar.slider("Petal width",0.1,2.5,1.3)

    to_dict={
        "sepal_length":sepal_length,
        "sepal_width":sepal_width,
        "petal_length":petal_length,
        "petal_width":petal_width
    }
    df=pd.DataFrame(to_dict,index=[0])
    return df

st.subheader("We want the category of this flower")

df=inputs_params()

df

iris=datasets.load_iris()

clf=RandomForestClassifier()

clf.fit(iris.data,iris.target)

category_predict=clf.predict(df)

st.subheader("The predicted category for the flower with above characteristics is:")
st.write(iris.target_names[category_predict])
