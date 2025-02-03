import streamlit as st
import pandas as pd
import networkx as nx


st.title("Let's build a food recommender system!")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data('restaurant-1-orders.csv')
st.write("The initial dataframe looks like this -")
st.write(df)


#df handling
@st.cache_data
def data_preprocesssing(df):
    dfd = df.drop_duplicates(subset=['Item Name'])
    df.drop(['Order Number', 'Quantity', 'Product Price', 'Total products'], axis=1, inplace=True)
    dfd.drop(['Order Number','Order Date', 'Quantity', 'Total products'], axis=1, inplace=True)
    df = df.groupby('Order Date', as_index=False).agg(
        {'Item Name': lambda x: ', '.join(x)}
    )
    dfd.reset_index(drop=True, inplace=True)
    df['Item Name'] = df['Item Name'].apply(lambda x: [dfd.loc[dfd['Item Name'] == item].index[0].astype(int) for item in x.split(", ")])
    return dfd, df

dfd, df = data_preprocesssing(df=df)
# -----------------------------------------------------------------

st.write("""After cleaning, processing, feature engineering, and grouping;
            The datafram looks like this -""")
st.write(df)

# -----------------------------------------------------------------

# complimentary recommendations

st.write("""## Let's now build a recommendation engine that can predict what dishes go best with the selected dish.""")

st.write("""For this problem statement we shall use a graph to capture
         the relationships between the dishes that are ordered together.""")

# Graph building

G = nx.empty_graph()

@st.cache_data
def populate_graph(df, _G):
    for _, row in df.iterrows():
        current_items = row["Item Name"]  # Get the list of items in the row

        # Fully connect all nodes in the current row
        for i, node1 in enumerate(current_items):
            for node2 in current_items[i+1:]:
                # Check if the edge already exists
                if G.has_edge(node1, node2):
                    # Increment the weight of the edge
                    G[node1][node2]['weight'] += 1
                else:
                    # Add a new edge with weight 1
                    G.add_edge(node1, node2, weight=1)

    self_loops = list(nx.selfloop_edges(G))
    print(f"Number of self-loops: {len(self_loops)}")
    if len(self_loops) > 0:
        print(f"Self-loops: {self_loops}")
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

G = populate_graph(df, G)

# get recommendations

def get_top_k_adjacent_nodes(graph, node, k):
    """
    Get the top k adjacent nodes with the highest edge weights for a given node.

    Args:
        graph (nx.Graph): The graph object.
        node: The node for which to find the top k neighbors.
        k (int): The number of top neighbors to retrieve.

    Returns:
        list: A list of tuples (neighbor, weight) sorted by weight in descending order.
    """
    if node not in graph:
        print(f"Node {node} is not in the graph.")
        return []
    
    # Get all neighbors of the node with their weights
    neighbors = [(neighbor, graph[node][neighbor]['weight']) for neighbor in graph.neighbors(node)]
    
    # Sort the neighbors by weight in descending order
    sorted_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
    
    # Return the top k neighbors
    return sorted_neighbors[:k]


# Example usage
node = 39  # The node to query
k = 10  # Top k neighbors
top_k_neighbors = get_top_k_adjacent_nodes(G, node, k)
st.write(f"Top {k} neighbors of node {node}: {top_k_neighbors}")


