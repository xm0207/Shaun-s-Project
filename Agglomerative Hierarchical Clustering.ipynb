# Load data into Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def read_names_into_dict():
    """
    Read company names into a dictionary
    """
    d = dict()
    with open("SP_500_firms.csv") as csvfile:
        input_file = csv.DictReader(csvfile)
        for row in input_file:
            #print(row)
            d[row['Symbol']] = [row['Name'],row['Sector']]
    return d

names_dict = read_names_into_dict()
comp_names = names_dict.keys()

# Read price data with pandas
filename = 'SP_500_close_2015.csv'
price_data = pd.read_csv(filename, index_col=0)
returns_data = price_data.pct_change()
print(returns_data.head())
returns_long = returns_data.stack().reset_index()
returns_long.columns = ['Date', 'Company', 'Daily_Return']

top_10_highest = returns_long.nlargest(10, 'Daily_Return')

top_10_lowest = returns_long.nsmallest(10, 'Daily_Return')


print("Top 10 Highest Daily Returns:")
print(top_10_highest)

print("\nTop 10 Lowest Daily Returns:")
print(top_10_lowest)
start_prices = price_data.iloc[0]
end_prices = price_data.iloc[-1]  
yearly_returns = (end_prices / start_prices) - 1
top_10_highest_yearly = yearly_returns.nlargest(10)
top_10_lowest_yearly = yearly_returns.nsmallest(10)


print("Top 10 Highest Yearly Returns:")
print(top_10_highest_yearly)

print("\nTop 10 Lowest Yearly Returns:")
print(top_10_lowest_yearly)
volatility = returns_data.std()

top_10_highest_volatility = volatility.nlargest(10)

top_10_lowest_volatility = volatility.nsmallest(10)

print("Top 10 Highest Yearly Volatilities:")
print(top_10_highest_volatility)

print("\nTop 10 Lowest Yearly Volatilities:")
print(top_10_lowest_volatility)
correlation_matrix = returns_data.corr()

print(correlation_matrix.head())
def top_correlated_companies(company, correlation_matrix, n):
    
    top_companies = correlation_matrix[company].sort_values(ascending=False)
    return top_companies[1:n+1]  

def bottom_correlated_companies(company, correlation_matrix, n):
    
    bottom_companies = correlation_matrix[company].sort_values(ascending=True)
    return bottom_companies[:n]  
tech_companies = ['AMZN', 'MSFT', 'FB', 'AAPL', 'GOOGL']

for company in tech_companies:
    print("Top 5 companies most correlated with "+company+": ")
    print(top_correlated_companies(company, correlation_matrix,5))

    print("Top 5 companies least correlated with "+company+": ")
    print(bottom_correlated_companies(company, correlation_matrix,5))
def create_correlation_list(correl):
    """
    Creates a list of correlations from a pandas dataframe of correlations
    
    Parameters:
        correl: pandas dataframe of correlations
    
    Returns:
        list of correlations containing tuples of form (correlation, ticker1, ticker2)
    """
    n_comp = len(correl.columns)
    comp_names = list(correl.columns)
    # Faster if we use a numpy matrix
    correl_mat = correl.to_numpy()
    L = [] # create list
    for i in range(n_comp):
        for j in range(i+1,n_comp):
            L.append((correl_mat[i,j],comp_names[i],comp_names[j]))
    return L

edges = create_correlation_list(correlation_matrix)
edges[:10]
# Load intermediary results from a "pickle" file
# You can use these with your algorithm below
import pickle
file_name = 'cluster_correlations'
with open(file_name, "rb") as f:
    correl = pickle.load(f)
    edges = pickle.load(f)

firms = list(correl.columns)
print(firms[:10])
edges[:10]
def find_bottom(node, next_nodes):
    """
    Find the "bottom" of a cluster starting from node in dictionary next_nodes

    Parameters:
        node: starting node
        next_nodes: dictionary of node connections

    Returns:
        the bottom node in the cluster
    """
    # Your code here

    while next_nodes[node] != node:
        node = next_nodes[node]
    return node
    
    pass


def merge_sets(node1, node2, next_nodes, set_starters):
    """
    Merges the clusters containing node1, node2 using the connections dictionary next_nodes.
    Also removes any bottom node which is no longer a "starting node" from set_starters.

    Parameters:
        node1: first node the set of which will be merged
        node2: second node the set of which will be merged
        next_nodes: dictionary of node connections
        set_starters: set of nodes that "start" a cluster

    Returns:
        does this function need to return something?
    """
    # Your code here
    bottom1 = find_bottom(node1, next_nodes)  
    bottom2 = find_bottom(node2, next_nodes)  

   
    if bottom1 == bottom2:
        return
    else:
        next_nodes[bottom2] = bottom1
        if bottom1 in set_starters:
          set_starters.remove(bottom1)



def cluster_correlations(edge_list, firms, k=200):
    """
    A mystery clustering algorithm
     
    Parameters:
         edge_list - list of edges of the form (weight,source,destination)
         firms - list of firms (tickers)
         k - number of iterations of algorithm

    Returns:
         next_nodes - dictionary to store clusters as "pointers"
            - the dictionary keys are the nodes and the values are the node in the same cluster that the key node points to
         set_starters - set of nodes that no other node points to (this will be used to construct the sets below)

    Algorithm:
         1 sort edges by weight (highest correlation first)
         2 initialize next_nodes so that each node points to itself (single-node clusters)
         3 take highest correlation edge
            check if the source and destination are in the same cluster using find_bottom
            if not, merge the source and destination nodes' clusters using merge_sets
         4 if max iterations not reached, repeat 3 with next highest correlation
         (meanwhile also keep track of the "set starters" ie nodes that have nothing pointing to them for later use)
    """
    # Sort edges
    sorted_edges = sorted(edge_list, key=lambda x: x[0], reverse=True)
    # Initialize dictionary of pointers
    next_nodes = {node: node for node in firms}
    # Keep track of "starting nodes", ie nodes that no other node points to in next_nodes
    set_starters = {node for node in firms}

    # Loop k times
    for i in range(k):
        # Your algorithm here
        weight, source, destination = sorted_edges[i]
       
        bottom_source = find_bottom(source, next_nodes)
        bottom_destination = find_bottom(destination, next_nodes)
        
        if bottom_source != bottom_destination:
          
          merge_sets(source, destination, next_nodes, set_starters)

   
    return set_starters, next_nodes
def construct_sets(set_starters, next_nodes):
    """
    Constructs sets (clusters) from the next_nodes dictionary
    
    Parameters:
        set_starters: set of starting nodes 
        next_nodes: dictionary of connections
    
    Returns: 
        dictionary of sets (clusters):
            key - bottom node of set; value - set of all nodes in the cluster
    
    """
    # Initialise an empty dictionary 
    all_sets = dict()
    
    # Loop:
    # Start from each set starter node
    # Construct a "current set" with all nodes on the way to bottom node
    # If bottom node is already a key of all_sets, combine the "current set" with the one in all_sets,
    # Otherwise add "current set" to all_sets
    for s in set_starters:
        cur_set = set()
        cur_set.add(s)
        p = s
        while next_nodes[p] != p:
            p = next_nodes[p]
            cur_set.add(p)
            
        if p not in all_sets:
            all_sets[p] = cur_set
        else: 
            for item in cur_set:
                all_sets[p].add(item)
    return all_sets
  starters,nodes=cluster_correlations(edges, list(price_data.columns), k=200)

clusters_200=construct_sets(starters,nodes)
def print_clusters_sorted_by_size(cluster_dict, top_n=10):
    sorted_clusters = sorted(cluster_dict.items(), key=lambda item: len(item[1]), reverse=True)
    
    sorted_clusters = sorted_clusters[:top_n]
    
    for cluster_id, members in sorted_clusters:
        members_list = ', '.join(members)
        print(f"Cluster starting with {cluster_id}: {members_list} (Total members: {len(members)})")

# Example usage
print_clusters_sorted_by_size(clusters_200)
for i in range (1,5):
    starters_set,nodes_set=cluster_correlations(edges, list(price_data.columns), k=100*i)
    clusters_i=construct_sets(starters_set,nodes_set)
    print("The result when k= "+str(i)+"00 is:")
    print_clusters_sorted_by_size(clusters_i)
#Q1:Try using sparse matrix to improve calculating efficiency
from scipy.sparse import coo_matrix

def build_sparse_matrix(edge_list, firms):
    
    firm_index = {firm: idx for idx, firm in enumerate(firms)}
    
    row = []
    col = []
    data = []

    for weight, source, destination in edge_list:
        row.append(firm_index[source])
        col.append(firm_index[destination])
        data.append(weight)

    sparse_matrix = coo_matrix((data, (row, col)), shape=(len(firms), len(firms)))
    
    return sparse_matrix, firm_index
def cluster_correlations_with_sparse(edge_list, firms, k=200):
    
    sparse_matrix, firm_index = build_sparse_matrix(edge_list, firms)

   
    next_nodes = {node: node for node in firms}

   
    set_starters = {node for node in firms}

    
    for i in range(min(k, sparse_matrix.nnz)):
        row, col = sparse_matrix.row[i], sparse_matrix.col[i]
        source = firms[row]
        destination = firms[col]

        bottom_source = find_bottom(source, next_nodes)
        bottom_destination = find_bottom(destination, next_nodes)

        if bottom_source != bottom_destination:
            merge_sets(source, destination, next_nodes, set_starters)

    return set_starters, next_nodes
starters_spare,nodes_spare=cluster_correlations_with_sparse(edges, list(price_data.columns), k=200)
clusters_spare200=construct_sets(starters,nodes)
print_clusters_sorted_by_size(clusters_spare200)
#Q2: Try to track the leader stock in cluster MTB (financial institutions) :

#specify MTB Stocks:
MTB_dic=clusters_spare200['MTB']

#deprive trading data in df_FI:
df_FI=returns_long[returns_long['Company'].isin(MTB_dic)]
#To find which stock changes most rapidly in return, and highlight it in BIG RED:

df_FI.loc[:, 'Return_Change'] = df_FI.groupby('Company')['Daily_Return'].diff()

leader_stock = df_FI.groupby('Company')['Return_Change'].mean().abs().sort_values(ascending=False).head(1)

print(f"Leader stock based on highest return change: {leader_stock.index[0]}")

#draw a trading diagram:
leader = leader_stock.index[0]
plt.figure(figsize=(10, 6))

leader_data = df_FI[df_FI['Company'] == leader]
plt.plot(leader_data['Date'], leader_data['Daily_Return'], label=f'Leader: {leader}', color='red', linewidth=2)

for company in df_FI['Company'].unique():
    if company != leader:
        company_data = df_FI[df_FI['Company'] == company]
        plt.plot(company_data['Date'], company_data['Daily_Return'], label=company, linestyle='--', alpha=0.6)

plt.title('Daily Return Comparison of Leader vs Other Stocks in Cluster by Volatility Comparisons')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
# Finding leader stock by individually carry out granger test
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import contextlib
import io

def find_best_leader(df, stock_list, target_stock, max_lag=1):
    best_stock = None
    best_pvalue = float('inf')
    
    target_data = df[df['Company'] == target_stock][['Date', 'Daily_Return']].rename(columns={'Daily_Return': target_stock}).set_index('Date')

    for candidate in stock_list:
        if candidate != target_stock:
            candidate_data = df[df['Company'] == candidate][['Date', 'Daily_Return']].rename(columns={'Daily_Return': candidate}).set_index('Date')
            test_data = pd.concat([target_data, candidate_data], axis=1).dropna()

            if not test_data.empty:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    try:
                        result = grangercausalitytests(test_data[[target_stock, candidate]], maxlag=max_lag)
                        p_value = result[max_lag][0]['ssr_ftest'][1]
                        if p_value < best_pvalue:
                            best_pvalue = p_value
                            best_stock = candidate
                    except Exception as e:
                        print(f"Granger test failed for {candidate} -> {target_stock}: {e}")
    
    return best_stock, best_pvalue

for stock in MTB_dic:
    best_leader, best_pvalue = find_best_leader(df_FI, MTB_dic, stock)
    print(f"The best leader stock for predicting {stock} is {best_leader} with a p-value of {best_pvalue}")
# Finding it using VAR model. The leader stock must have the lowest sum of p values and less maximum p value to predict future price of other stocks
import pandas as pd
from statsmodels.tsa.api import VAR
import numpy as np

def prepare_data(df, leader_stock, target_stocks):
    leader_data = df[df['Company'] == leader_stock][['Date', 'Daily_Return']].rename(columns={'Daily_Return': leader_stock})
    leader_data['Date'] = pd.to_datetime(leader_data['Date'])
    leader_data = leader_data.set_index('Date')

    target_data = pd.DataFrame()

    for stock in target_stocks:
        stock_data = df[df['Company'] == stock][['Date', 'Daily_Return']].rename(columns={'Daily_Return': stock})
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data = stock_data.set_index('Date')

        target_data = pd.concat([target_data, stock_data], axis=1)

    combined_data = pd.concat([leader_data, target_data], axis=1).dropna()
    return combined_data

def granger_for_leader(df, leader_stock, target_stocks, max_lag=1):
    data = prepare_data(df, leader_stock, target_stocks)
    model = VAR(data)
    results = model.fit(maxlags=max_lag)

    granger_result = {}
    for target in target_stocks:
        p_value = results.test_causality(causing=leader_stock, caused=target, kind='f').pvalue
        granger_result[target] = p_value

    return granger_result

def find_best_leader(df, stock_list, max_lag=1):
    best_leader = None
    best_pvalue_sum = float('inf')
    best_max_pvalue = None

    for stock in stock_list:
        target_stocks = [s for s in stock_list if s != stock]
        granger_result = granger_for_leader(df, stock, target_stocks, max_lag)
        pvalue_sum = sum(granger_result.values())
        max_pvalue = max(granger_result.values())

        if pvalue_sum < best_pvalue_sum:
            best_pvalue_sum = pvalue_sum
            best_leader = stock
            best_max_pvalue = max_pvalue

    return best_leader, best_pvalue_sum, best_max_pvalue

best_leader, best_pvalue_sum, best_max_pvalue = find_best_leader(df_FI, MTB_dic)

print(f"The best leader stock is {best_leader} with a total p-value sum of {best_pvalue_sum} and a maximum p-value of {best_max_pvalue}")
leader_data = df_FI[df_FI['Company'] == best_leader]
plt.plot(leader_data['Date'], leader_data['Daily_Return'], label=f'Leader: {best_leader}', color='red', linewidth=2)

for company in df_FI['Company'].unique():
    if company != best_leader:
        company_data = df_FI[df_FI['Company'] == company]
        plt.plot(company_data['Date'], company_data['Daily_Return'], label=company, linestyle='--', alpha=0.6)

plt.title('Daily Return Comparison of Leader vs Other Stocks in Cluster by VAR Model')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
