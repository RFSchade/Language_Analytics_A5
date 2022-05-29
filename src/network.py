#=====================================================#
#=============> Network Analysis Script <=============#
#=====================================================#

#=====> Import modules
# System tools
import os

# Data analysis
import pandas as pd
from tqdm import tqdm
import random
from datetime import datetime

# Network analysis tools
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

# Useability tools
import argparse

#====> Define functions
# > Load edge data
def load_data(filename, cleaning = True):  
    if cleaning == True:
        # Get the filepath
        filepath = os.path.join("in", filename)
        # Reading the filepath 
        with open(filepath, "r") as f:
            txt = f.read()
    
        # > Get rid of hastaged lines 
        # Split on newline
        split_txt = txt.split("\n")
        # Get rid of all lines containing a hastag
        txt_data = list(filter(lambda x: "#" not in x, split_txt))
        # Join data again
        txt_data = '\n'.join(txt_data)
    
        # > Write new file without hastags
        # Define outpath 
        outpath = os.path.join("in", "clean_edges.txt")
        with open(outpath, "w") as f:
            f.write(txt_data)
            
        # > Load data without hastag  
        # Get the filepath
        filepath = os.path.join("in", "clean_edges.txt")
        # Reading the filepath 
        data = pd.read_csv(filepath, header = None, sep='\t', names = ("Source", "Target"))
    else:
        # > Load data without hastag  
        # Get the filepath
        filepath = os.path.join("in", filename)
        # Reading the filepath 
        data = pd.read_csv(filepath, header = None, sep='\t', names = ("Source", "Target"))
    
    # Convert columns from int to string
    data["Source"] = data["Source"].apply(str)
    data["Target"] = data["Target"].apply(str)
    
    return data

# > Load date data
def load_dates(filename, cleaning = True):  
    if cleaning == True:
        # Get the filepath
        filepath = os.path.join("in", filename)
        # Reading the filepath 
        with open(filepath, "r") as f:
            txt = f.read()
    
        # > Get rid of hastaged lines 
        # Split on newline
        split_txt = txt.split("\n")
        # Get rid of all lines containing a hastag
        txt_data = list(filter(lambda x: "#" not in x, split_txt))
        # Join data again
        txt_data = '\n'.join(txt_data)
    
        # > Write new file without hastags
        # Define outpath 
        outpath = os.path.join("in", "clean_dates.txt")
        with open(outpath, "w") as f:
            f.write(txt_data)
            
        # > Load data without hastag  
        # Get the filepath
        filepath = os.path.join("in", "clean_dates.txt")
        # Reading the filepath 
        dates = pd.read_csv(filepath, header = None, sep='\t', names = ("ID", "date"))
    else:
        # > Load data without hastag  
        # Get the filepath
        filepath = os.path.join("in", filename)
        # Reading the filepath 
        dates = pd.read_csv(filepath, header = None, sep='\t', names = ("ID", "date"))
    
    # Convert columns from int to string
    dates["ID"] = dates["ID"].apply(str)
    dates["date"] = pd.to_datetime(dates["date"], format = "%Y-%m-%d")
    
    return dates   

# > Sample data
def sample(data, seed = 42, min_freq = 10, nr_nodes = 100):
    # Get dataframe of the nr. of appearances of each ID
    count_df = data.apply(pd.value_counts)
    # Filter out atricles that do not appear often 
    filter_list = count_df[count_df["Source"] > 10].index.tolist()
    filter_df = data[data["Source"].isin(filter_list)]
    
    # Set seed 
    random.seed(seed)
    # Get list of source nodes
    node_list = filter_df["Source"].unique().tolist()
    # Get a random sample of source nodes
    sample_nodes = random.sample(node_list, nr_nodes)
    # Keep only the edges that involve those nodes
    sample_df = data[data["Source"].isin(sample_nodes) | data["Target"].isin(sample_nodes)]
    
    return sample_df

# > Create figure 
def make_figure(data):
    # Print info 
    print("[INFO] Creating graph...")
    # Define directed graph
    G = nx.from_pandas_edgelist(data, source='Source', target='Target', edge_attr=None, create_using=nx.DiGraph())
    # Draw figure 
    nx.draw_networkx(G, with_labels=False, node_size=20)
    # Define outpath
    outpath = os.path.join("output", "network_graph.png")
    # Save figure 
    plt.savefig(outpath, dpi=100, bbox_inches="tight")
    
    return G

# > Get centrality scores
def centrality_scores(G, dates):
    # Print info
    print("[INFO] Calculating centrality scores...")
    
    # Finding degrees and creating dataframe 
    degrees = G.degree()
    df = pd.DataFrame(degrees, columns = ["ID", "degree"])
    # Finding in degree
    in_degrees = G.in_degree()
    df["in_degree"] = [v for k, v in in_degrees]
    # Finding out degree
    out_degrees = G.out_degree()
    df["out_degree"] = [v for k, v in out_degrees]
    # Finding and adding betweenness centrality 
    bc = nx.betweenness_centrality(G)
    df["betweenness"] = bc.values()
    # Finding and adding eigrnvector centrality
    ev = nx.eigenvector_centrality(G)
    df["eigenvector"] = ev.values()
    
    # > Add dates for good measure
    merged = pd.merge(df, dates, on="ID")

    # Save the dataframe
    outpath = os.path.join("output", "centrality_df.csv")
    merged.to_csv(outpath, index=False)
    
    return merged

# > Create reading list 
def get_reading_list(scores):
    # Get articles with 10 highest degree centrality
    high_degree = scores.nlargest(10, "degree")
    # Get articles with 10 highest in degree centrality
    high_in = scores.nlargest(10, "in_degree")
    # Get articles with 10 highest out degree centrality
    high_out = scores.nlargest(10, "out_degree")
    # Get articles with 10 highest betweenness centrality
    high_bc = scores.nlargest(10, "betweenness")
    # Get articles with 10 highest eigenvector centrality
    high_ev = scores.nlargest(10, "eigenvector")
    # Add it all together 
    reading_list = pd.concat([high_degree, high_in, high_out, high_bc, high_ev])
    # Remove duplicates
    reading_list = reading_list.drop_duplicates() 
    
    # Save the dataframe
    outpath = os.path.join("output", "reading_list.csv")
    reading_list.to_csv(outpath, index=False)
    
    return reading_list

# Plot date histogram 
def histogram(scores):
    # Define figure
    fig, ax = plt.subplots()
    # Plot histogram
    ax.hist(scores["date"], bins = 50)
    # Add title
    plt.title("Date histogram", fontsize = 15)
    # Save plot
    plt.savefig(os.path.join("output", "histogram_img.png"), bbox_inches = "tight")
    
# > Prep data for plotting 
def plot_prep(scores):
    # Create dateobject that contains year and month, but not day 
    scores['month'] = scores['date'].apply(lambda x: x.strftime('%Y-%m'))
    # Calculate mean scores by month
    mean_df = scores.drop(["ID", "date"], axis=1).groupby("month").mean().reset_index()
    # Ensure that "month" is still a datetime type 
    mean_df["month"] = pd.to_datetime(mean_df["month"], format = "%Y-%m")
    
    return mean_df

# > Plot degree centrality 
def plot_degrees(data):
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)
    
    # > Draw subplot 1 (degrees)
    ax1.set_title("Degrees", fontsize = 15)
    ax1.set_ylabel("Mean degrees (by month)", fontsize = 12)
    ax1.bar(data["month"], data["degree"], width = 30)
    # > Draw subplot 2 (in degrees)
    ax2.set_title("In degrees", fontsize = 15)
    ax2.set_ylabel("Mean in degrees (by month)", fontsize = 12)
    ax2.bar(data["month"], data["in_degree"], width = 30)
    # > Draw subplot 3 (out degrees)
    ax3.set_title("Out degrees", fontsize = 15)
    ax3.set_ylabel("Mean out degrees (by month)", fontsize = 12)
    ax3.bar(data["month"], data["out_degree"], width = 30)
    
    # Save plot
    plt.savefig(os.path.join("output", "degree_img.png"), bbox_inches = "tight")
    
# > Plot betweenness and eigenvector centrality 
def bc_ev_plot(data): 
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    # Draw bars
    ax1.set_title("Betweenness Centrality", fontsize = 15)
    ax1.set_ylabel("Mean BC (by month)", fontsize = 12)
    ax1.bar(data["month"], data["betweenness"], width = 30)
    ax2.set_title("Eigenvector Centrality", fontsize = 15)
    ax2.set_ylabel("Mean in EVC (by month)", fontsize = 12)
    ax2.bar(data["month"], data["eigenvector"], width = 30)

    # Save plot
    plt.savefig(os.path.join("output", "bc_ev_img.png"), bbox_inches = "tight")
    
# > Parse arguments
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-e", "--edges", 
                    required=False, 
                    help="Name of txt file containing edges", 
                    default="cit-HepPh.txt")
    ap.add_argument("-d", "--dates", 
                    required=False, 
                    help="Name of txt file containing dates", 
                    default="cit-HepPh-dates.txt")
    ap.add_argument("-nc", "--no_cleaning", 
                    action='store_false',
                    help="Add argument if data does not need cleaning",
                    required=False)
    ap.add_argument("-s", "--seed", 
                    required=False, 
                    type=int,
                    help="Set random seed for random sampling", 
                    default=42)
    ap.add_argument("-min", "--min_freq", 
                    required=False,
                    type=int,
                    help="Minimum nr. of times an article appears in the dataset for it to be included in analysis",
                    default=10)
    ap.add_argument("-nr", "--nr_nodes", 
                    required=False,
                    type=int,
                    help="Nr. of nodes to be chosen for the sample",
                    default=100)
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args

#=====> Define main()
def main(): 
    # Get arguments
    args = parse_args()
    
    # Print info
    print("[INFO] Loading data...")
    
    # Load edge data
    data = load_data(args["edges"], args["no_cleaning"])
    # Load dates
    dates = load_dates(args["dates"], args["no_cleaning"])
    # Sample data 
    sample_df = sample(data, args["seed"], args["min_freq"], args["nr_nodes"])
    
    # Make figure
    G = make_figure(sample_df)
    # Get centrality scores
    scores = centrality_scores(G, dates)
    # Get reading list 
    reading_list = get_reading_list(scores)
    
    # Print info
    print("[INFO] Plotting...")
    
    # Plot histogram
    histogram(scores)
    # Prep data
    plot_data = plot_prep(scores)
    # Plot centrality measures
    plot_degrees(plot_data)
    bc_ev_plot(plot_data)
    
    # Print info
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()
    
