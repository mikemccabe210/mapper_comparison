# Mapper Comparison

Accompanying code for paper *Mapper Comparison with 
Wasserstein Metrics*.

This repo is intended for the evaluation of different candidate metrics for 
comparing Mapper graphs. Mapper is an unsupervised learning algorithm based 
in Topological Data Analysis that generates a simplified nerve complex 
representation from a data set that captures topological features of a 
metric space.

![alt text](images/mapper_overview.png)

One challenge using Mapper graphs is that without an indicator of goodness
of fit, it can be challenging to determine when a Mapper model (or nerve 
complex representations in general) no longer fits that data. However, if 
we had a metric over Mapper graphs that captured all of the properties we're
interested in, then we could evaluate the degree to which Mapper graphs would
be expected to change over different samples of the same population, which 
could be used to identify when new samples have significantly deviated from
the previous model. 

This repository contains code from the paper *Mapper Comparison with 
Wasserstein Metrics*, which attempts to address this issue by comparing Mapper
graphs as metric-measure spaces and introduces a Wasserstein distance variant,
 the Network Augmented Wasserstein (NAW) distance, which intuitively captures
 differences between the topological, metric, and density information represented
 by Mapper graphs. 
 
## Usage

This repo comes equipped with the actual distance measures and with a lightweight 
mapper implementation that the distance measures are built around. The mapper builder
objects automatically create a networkx object that can be used for visualization. The
basic procedure for generating a network and using a network metric is:

```python
from lightweight_mapper.clustering import SingleLinkageHistogram
from lightweight_mapper.networks import Network, Cover, Partition
from lightweight_mapper.partitioners import UniformPartitioner
from distances import naw_distance

metric_space = pairwise_distances(X) # Create finite metric space
partition = UniformPartitioner() # Create partitioner
cover1 = partition.fit_transform(X[:, 0], 130, .7, 'PCA1') # Define cover parameters
cover2 = partition.fit_transform(X[:, 1], 130, .7, 'PCA2') # Define cover parameters
cover = [cover1, cover2] # Create cross product cover
cluster = SingleLinkageHistogram(metric='precomputed', threshold = 'histogram') # Any sklearn API compatible clusterer

network1 = Network(metric_space, cover, cluster, prune=True)

...

distance = naw_distance(network1, network2, metric_space)
```


## Examples
Examples can be found in [examples](https://github.com/mikemccabe210/mapper_comparison/) (Eventually)

### References

Michael McCabe. Mapper Comparison with Wasserstein Metrics. arXiv preprint arXiv:1812.06232, 2018. [[arxiv]](https://arxiv.org/abs/1812.06232)

