import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import streamlit.components.v1 as components
from PIL import Image
components.html("""
                    <h3 style="text-align: justify;color:rgb(0, 163, 108)">
                     Flower iris category prediction App </h3>
                     
                     <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9Gc
                     ThHqOVOAE_IsPyWUJRTBNMmG9RCllDeJt8wQ&usqp=CAU" />
                """

                )

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

import streamlit.components.v1 as components

components.html("""
                    <h2 style="text-align: center"> Data exploration </h2>
                
                """
                )