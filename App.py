import streamlit as st
import joblib
import pandas as pd

def recommend_products(product_id, similarity_df, top_n=5):
    if product_id in similarity_df.columns:
        similar_scores = similarity_df[product_id].sort_values(ascending=False)[1:top_n+1]
        return similar_scores.index.tolist()
    else:
        return []

st.title("Customer Segmentation & Product Recommendation")

menu=st.sidebar.selectbox("Choose Module",["Product Recommendation", "Customer Segmentation"])

if menu=="Product Recommendation":
    st.header("Product Recommendation")
    prod_input=st.text_input("Enter the Product name")
    if st.button("Get Recommendations"):
        sim_matrix=joblib.load("Downloads/Guvi/Project/Projects List/Project 4 - Recommendation System/product_similarity.pkl")
        try:
            results=recommend_products(prod_input,sim_matrix)
            st.write("### Recommended Products:")
            for i,p in enumerate(results):
                st.write(f" {i+1}.  {p}")
        except:
            st.error("Product not found")

elif menu == "Customer Segmentation":
    st.header("Customer Segmentation")
    r=st.number_input("Recency (days)", min_value=0)
    f=st.number_input("Frequency",min_value=0)
    m=st.number_input("Monetary", min_value=0)
    if st.button("Predict Cluster"):
        model=joblib.load('Downloads/Guvi/Project/Projects List/Project 4 - Recommendation System/kmeans_model.pkl')
        std_scaler=joblib.load("Downloads/Guvi/Project/Projects List/Project 4 - Recommendation System/scaler.pkl")
        input_data=std_scaler.transform([[r,f,m]])
        cluster=model.predict(input_data)[0]
        segment=["Occasional","At-Risk","Regular","High Value"][cluster]
        st.success(f"Predicted Cluster: {segment}")







        