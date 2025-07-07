# GlaSEE: Glacier Snow mapping with Earth Engine

[Rainey Aberle](https://github.com/RaineyAbe), [Jukes Liu](https://github.com/jukesliu), and [Ellyn Enderlin](https://github.com/ellynenderlin)

[CryoGARS Glaciology](https://github.com/CryoGARS-Glaciology)

Department of Geosciences, Boise State University

## Correspondence

Rainey Aberle (raineyaberle@u.boisestate.edu)

## Description

Pipeline for land cover classification in Sentinel-2 and Landsat 8/9 imagery using machine learning classifiers trained using manually-classified points at the U.S. Geological Survey Benchmark Glaciers. 

This repository is the partner to [`glacier-snow-cover-mapping`](https://github.com/RaineyAbe/glacier-snow-cover-mapping). Here, rather than classifying and exporting images locally, all computations are done on the Google Earth Engine server and summary statistics are exported to your Google Drive. 

## Installation

1. Clone (or fork and then clone) this repository.

```
$ git clone https://github.com/RaineyAbe/glasee
```

2. Install the required packages listed in `environment.yml`. We recommend using Mamba or Micromamba for environment management. 

```
$ cd glasee
$ micromamba env create -f environment.yml
```

3. If using Jupyter for running notebooks, add the environment as an ipykernel:

```
$ micromamba activate glasee
$ python -m ipykernel install --name glasee
```

## Requirements

1. Google Earth Engine (GEE) account: used to query imagery and the DEM (if no DEM is provided). Sign up for a free account [here](https://earthengine.google.com/new_signup/). 

2. Google Drive folder for exports.

## Examples

The pipeline can be run for a single site (`snow_classification_pipeline_single-site.ipynb`) or multiple sites at once (`snow_classification_pipeline_multi-site.ipynb`). 

Snow cover statistics for each site will be exported in multiple CSV files. To compile the CSV files for each site and remove any empty files, see the `post_process.ipynb` notebook. 

Please refer to each notebook for more information.

## Notes on GEE job submissions

GEE enforces [user quotas](https://developers.google.com/earth-engine/guides/usage) on memory usage (default = 10 MB per query) and the number of concurrent requests (default = 40). The GlaSEE pipeline mitigates exceeding these limits by splitting tasks into smaller date ranges and increasing the image scale only if necessary.

The date range specified by the user is split into smaller ranges before exporting based on the area of the glacier (AOI):

- AOI < 500 km<sup>2</sup>: split by month
- 700 km<sup>2</sup> <= AOI < 1100 km<sup>2</sup>: split by week
- AOI >= 1100 km<sup>2</sup>: split by day

For glaciers with areas > 3000 km<sup>2</sup>, splitting the date range by day is often not enough to avoid computation time out. Therefore, the images are automatically upscaled to 200 m resolution, which we found empirically to work for these largest glaciers.

Especially for the largest glaciers, many exports may be empty, depending on image availability. To compile all results and remove empty CSVs from your Google Drive folder, see the `post_processing.ipynb` notebook in this repo. 

## Citation

Please reference the following when using or presenting this work:

Aberle, R., Enderlin, E., O’Neel, S., Florentine, C., Sass, L., Dickson, A., et al. (2025). Automated snow cover detection on mountain glaciers using spaceborne imagery and machine learning. The Cryosphere, 19(4), 1675–1693. https://doi.org/10.5194/tc-19-1675-2025

## Acknowledgements

This work was funded by BAA-CRREL (award no. W913E520C0017), NASA EPSCoR (award no. 80NSSC20M0222), the NASA Idaho Space Grant Consortium summer internship program, and the SMART (Science, Mathematics, And Research for Transformation) Scholarship-for-Service program. This research was supported by the U.S. Geological Survey Ecosystem Mission Area Climate Research and Development Program.