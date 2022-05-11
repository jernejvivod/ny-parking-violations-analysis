# General Instructions

Adhere to the CRISP-DM methodology.

# Tools

Implement the functionality in a proper distributed manner (Dask + SLURM)

# Data

The initial dataset is the New York City open data - Parking Violations that you will augment
with additional information from the data sources of your choice (as per instructions below)

In all cases, the analysis should focus on:
- Full data
- Boroughs
- An "interesting" subset of streets (e.g. most problematic streets)

# Tasks

Your project must consist of the following tasks:

## Task 1

Import CSV datasets and store them in:
- Parquet format
- Avro format
- HDF5 format

Use these three formats in subsequent work.

Start with Parquet, then use Avro and HDF5. Compare the datasets in terms of file sizes.
Chose appropriate partitioning where applicable.

## Task 2

Augment the original Parking violations data with sources of additional information:
- Weather information
- Vicinity/locations of primary and high schools
- Information about events in vicinity
- Vicinity/locations of major businesses
- Vicinity/locations of major attractions


You can find man (but not all) sources in the New York City open data repository. You will
need to link the data with respect to location (location data or street names) and time (where
applicable, e.g. for weather and event data).


## Task 3

Perform the introductory exploratory data analysis. Select and calculate appropriate data
aggregates. Determine and visualize how good your data augmentation is.


## Task 4

Perform the data analysis in a "streaming" manner (treat data as streaming).
Show rolling descriptive statistics (mean, standard deviation, ..., think of at least three more) for all data, boroughs,
and for 10 most interesting streets (with highest numbers of tickets overall, or by your qualified choice).
For the same data, implement, or apply a stream clustering algorithm of your choice.

## Task 5

Perform the data analysis in a "batch" manner using machine learning to predict events such as
days with a high number of tickets (think of and implement at least one additional learning problem).
You will need to appropriately transform the augmented data.
Ensure that the workers will not have enough memory to store and process the entire dataset
(e.g., 8GB per worker). Use at least three kinds of supervised machine learning algorithms:
- One of the simple distributed algorithms from Dask-ML
- a complex third-party algorithm which "natively" supports distributed computing (XGBoost, LightGBM, ...)
- One of the common scikit-learn algorithms utilizing partial\_fit.

For all these scenarios, compare performance in terms of loss (error), scalability, time, and total
memory consumption.


# Reporting

Prepare a PDF report of at least 10 pages (the code must also be submitted). Each team will present
the results in a short 5-minute presentation at the end of the semester.

# Data folder on Arnes HPC Cluster

`/d/hpc/projects/FRI/bigdata/data/NYTickets`

# Link to Dataset

https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2022/pvqr-7yc4

# Useful Links

CRISP-DM: http://lyle.smu.edu/~mhd/8331f03/crisp.pdf
