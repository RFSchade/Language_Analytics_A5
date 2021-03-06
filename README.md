# Language Analytics Assignment 5
## Assignment description
This repository contains my submission of the final portfolio assignment for the Language Analytics course at Aarhus university, the topic of which is self-assigned. 
My goal for this assignment is to write a program that can provide a quick overview of a field of scientific literature using a network analysis of article citations (this analysis is done using the [network]( https://pypi.org/project/networkx/) python module).     
While, ideally, I envision this tool used in combination with web scraping (so a potential user perhaps could find a place to start when trying to get an understanding of a new field), but as web scraping was outside the scope of the project, analysis was carried out on the High-energy physics citation network (cit-HepPh), a large dataset of 34,546 physics papers with 421,578 edges (full citation can be found at the end of the readme file).     
Dataset can be sourced here:     
https://snap.stanford.edu/data/cit-HepPh.html    
## Methods
The script loads the data, cleans it of hash-tagged lines, and then samples the data by first excluding papers that do not show up in many edges, and the by using random node sampling (though this is not the optimal way of doing it – according to (Leskovec & Faloutsos, 2006),  a random walk algorithm would have been preferable, though it was, again, outside the scope of this project).     
The script then creates a network graph using the sampled data and calculates the following centrality scores: Degree centrality (including in and out degrees), betweenness centrality, and eigenvector centrality.     
After this, the script compiles a “reading list” of the papers with the highest centrality metrics (note: while not ideal, cit-HepPh does not contain the titles or authors of the papers, and as such cannot be identified, but, ideally, this information should be available).     
The script also plots a histogram of dates the papers were published, and 5 plots of the centrality scores averaged across months.      

## Repository structure
in: Folder for input data    
notebooks: Folder for experimental code    
output: Folder for the output generated by the scrips – at present it contains 4 files:    
- bc_ev_img.png:
    - Output of the network.py script – an image of two figures that plots the mean monthly eigenvector and betweenness centrality scores across the analyzed data respectively.
- centrality.csv:
    - Output of the network.py script – a dataframe of 7 columns and 2912 rows. The “ID” column contains the IDs of the papers in the analysis. The next 6 columns contain the paper’s the degree, in degree, out degree, betweenness and eigenvector centrality. The last column contains the time the paper was submitted to [Arxiv]( https://arxiv.org/).
-	degree_img.png:
    - Output of the network.py script – an image containing three graphs that plot the average monthly degree, in degree, and out degree centrality across the analyzed data.
- histogram_img.png:
    - Output of the network.py script – contains a histogram of the time of submission to Arxiv for the papers included in analysis.
-	network_graph.png: 
    - Output of the network.py script – contains a graph of the directed network of citations generated in analysis.
-	reading_list.csv: 
    - Output of the network.py script – a dataframe that gathers the centrality scores (as in centrality.csv) of the papers with the 10 highest of each type of centrality scores (which is the filtered for duplicates)

src: Folder for python scripts    
- \_\_init__.py
- network.py

github_link.txt: link to github repository    
requirements.txt: txt file containing the modules required to run the code    

## Usage
Modules listed in requirements.txt should be installed before the script is run.    
__Note about requirements to run the script__    
It is important that both the network and scipy are up to date – if this is true and there is still issues, try uninstalling and downgrading network to networkx==2.6.3    
__Input data__
The format of the input data should be an edgelist stored in a tap delimited txt file and another tap delimited file containing the ID of all the articles in the edgelist and the time they were submitted to Arxiv.     

__network.py__    
To analyze the data, run network.py from the Language_Analytics_A5 repository folder. The script has six arguments:    
-	_-e or --edges: Name of txt file containing edge data – default is "cit-HepPh.txt"_
-	_-d or --dates: Name of txt file containing date data – default is “cit-HepPh-dates.txt”_
-	_-nc or --no_cleaning: Argument that decides if data should go through the process of being cleaned from comment lines - add argument if data does not need cleaning. if argument is added, the clean edge (-e) and date (-d) data should also be specified_
-	_-s or --seed: Random seed for random sampling of data – default is 42_
-	_-min or --min_freq: Minimum nr. of times an article appears in the dataset for it to be included in analysis – default is 10_ 
-	_-nr or --nr_nodes: Nr. of nodes to be chosen randomly for the sample – default is 100_

Example of code running the script from the terminal:    
```
python src/network.py -nc -e edge_data.txt -d date_data.txt -s 42 -min 10 -nr 100
```

## Discussion of results
I ran the script with all the arguments at default.     
network_graph.png looks something like a pupil with an iris around it. In the “pupil” there are several different close-knit clusters of papers citing each other. At the edges of the “iris”, you find individual papers that do not appear to by cited more than once or twice by the papers in the “pupil”, and they do not cite any other articles on the graph themselves. This could be taken as an indication that the papers in the iris are less relevant to the issues being focused on in the other articles. On the other hand, this might be an artifact of the sampled data – perhaps these seemingly isolated papers are a part of their own cluster, if you were to analyze a greater amount of data, or even if the sampling algorithm had chosen different nodes to include?     
In degree_img.png, the bulk degree centrality appears to increase in sudden spikes the more recent the papers are – in degree centrality appear to be higher in earlier texts, and out degree in later texts. This is intuitive, as earlier texts would have more time to be cited, and later texts would have more papers to cite as the body of work within the field grows (as seen in histogram_img.png).    
The same pattern can be seen in betweenness and eigenvector centrality. This, again seems intuitive – as a field grows larger, it becomes more difficult to read all papers published within it – and a such, a niche appears for papers that can summarize large bodies of the literature (perhaps a meta-study?) and these papers are likely to have a higher betweenness centrality, as they enable authors to use them as a “link” to the points made in the papers they summarize.
This also makes it likely for the eigenvector centrality to rise, as more articles with manny connections link to each other.    
though both graphs in bc_ev_img.png seem sparser than those found in degree_img.png – this could again be an artifact of the sampling method. It makes sense that the betweenness and eigenvector centrality metrics are more impacted by potential clusters being cut up in sampling, as both these measures look at the influence of nodes in the broader context of the network.     

## Citations
-	J. Leskovec, J. Kleinberg and C. Faloutsos. Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2005.
-	J. Gehrke, P. Ginsparg, J. M. Kleinberg. Overview of the 2003 KDD Cup. SIGKDD Explorations 5(2): 149-151, 2003.
-	Leskovec, J., & Faloutsos, C. (2006). Sampling from large graphs. Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining  KDD ’06, 631. https://doi.org/10.1145/1150402.1150479

