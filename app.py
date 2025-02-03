import streamlit as st
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

# Streamlit app title
st.title("🍽️ The Food Recommender: What Goes Best With Your Meal?")

st.write("""
Imagine walking into your favorite restaurant. You’re excited, but the menu is overwhelming. 
Wouldn’t it be great if you could get recommendations not just for what goes well together, 
but also for price-friendly alternatives? Let's build a smart food recommender system to do just that!
""")

# Load and cache data
@st.cache_data
def load_data():
    return pd.read_csv('restaurant-1-orders.csv')

df = load_data()
st.write("### 📌 Step 1: Understanding the Raw Data")
st.write("Here’s a glimpse of the raw order data:")
st.dataframe(df.head())

# Data Preprocessing
@st.cache_data
def preprocess_data(df):
    dfd = df.drop_duplicates(subset=['Item Name'])
    df.drop(['Order Number', 'Quantity', 'Product Price', 'Total products'], axis=1, inplace=True)
    dfd.drop(['Order Number','Order Date', 'Quantity', 'Total products'], axis=1, inplace=True)
    
    df = df.groupby('Order Date', as_index=False).agg(
        {'Item Name': lambda x: ', '.join(x)}
    )
    
    dfd.reset_index(drop=True, inplace=True)
    df['Item Name'] = df['Item Name'].apply(lambda x: [dfd.loc[dfd['Item Name'] == item].index[0] for item in x.split(", ")])
    return dfd, df

dfd, df = preprocess_data(df)

st.write("### 📌 Step 2: Processed Data Ready for Predictions")
st.write("After cleaning and feature engineering, here’s the refined dataset:")
st.dataframe(df.head())

# Building the Co-Ordering Graph
st.write("## 🍜 Step 3: Understanding Food Pairings with Graphs")
st.write("""
Customers tend to order dishes together. Let’s capture this relationship by 
building a network where nodes represent dishes, and edges represent how frequently 
these dishes are ordered together. The stronger the edge, the more often they are paired.
""")

# Graph building
G = nx.Graph()

@st.cache_data
def build_graph(df, _G):
    for _, row in df.iterrows():
        items = row["Item Name"]
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                if G.has_edge(item1, item2):
                    G[item1][item2]['weight'] += 1
                else:
                    G.add_edge(item1, item2, weight=1)

    return G

G = build_graph(df, G)
st.write(f"🔗 **Graph Built!** It contains **{G.number_of_nodes()}** dishes and **{G.number_of_edges()}** connections.")

# Getting top co-ordered dishes
st.write("### 🔥 Find the Best Combinations!")
selected_dish = st.selectbox("Choose a dish to see what goes best with it:", dfd["Item Name"].tolist())

def get_top_k_combinations(G, item, k):
    if item not in G:
        return []
    neighbors = [(neighbor, G[item][neighbor]['weight']) for neighbor in G.neighbors(item)]
    return sorted(neighbors, key=lambda x: x[1], reverse=True)[:k]

if selected_dish:
    dish_index = dfd[dfd["Item Name"] == selected_dish].index[0]
    recommendations = get_top_k_combinations(G, dish_index, 5)

    st.write(f"🥗 **Top 5 dishes frequently ordered with {selected_dish}:**")
    for dish, weight in recommendations:
        st.write(f"- {dfd.iloc[dish]['Item Name']} (Ordered together {weight} times)")

# Price-Based Alternative Recommendations
st.write("## 💰 Step 4: Finding Price-Friendly Alternatives")
st.write("""
What if you want something similar but cheaper (or more premium)? 
Let's use KNN to suggest dishes with similar pricing!
""")

# Fit KNN model
prices = dfd["Product Price"].values.reshape(-1, 1)
knn = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(prices)

def find_price_alternatives(item, dfd, knn):
    item_price = dfd.loc[dfd["Item Name"] == item, "Product Price"].values[0].reshape(-1, 1)
    distances, indices = knn.kneighbors(item_price)
    return [(dfd.iloc[idx]["Item Name"], dfd.iloc[idx]["Product Price"]) for idx in indices[0]]

if selected_dish:
    alternatives = find_price_alternatives(selected_dish, dfd, knn)
    st.write(f"💲 **Top 5 price-friendly alternatives for {selected_dish}:**")
    for alt_dish, price in alternatives:
        st.write(f"- {alt_dish} (Price: ${price:.2f})")

st.write("### 🎯 Conclusion")
st.write("""
With this recommender, you can now make informed food choices! Whether you want 
to find the perfect pairing or look for budget-friendly alternatives, our model 
helps make dining decisions smarter and more enjoyable! 🍽️
""")