# Repository Finder

## Overview
Repository Finder is a tool that identifies and analyzes open-source repositories affiliated with universities using GitHub metadata and contributor analysis. The institutions covered by this project are members of [CURIOSS](https://curioss.org/) (Community for University and Research Institution OSPOs). The pipeline is split into four modular scripts:

- **main_scraping.py**: Fetches and stores raw repository, organization, and contributor data.

- **main_filtering.py**: Filters repositories based on affiliation with a specific university.

- **main_analysis.py**: Analyzes and visualizes filtered repository data, including license usage, language distribution, and community practices.

- **main_analysis_combined.py**: Analyzes and visualizes aggregated repository data by type. We use this script to analyze data across all 32 supported institutions.

## Installation
1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up GitHub API access:**
   - Create a `.env` file in the root directory and add:
     ```
     GITHUB_TOKEN=your_personal_access_token
     OPENAI_API_KEY=your_openai_token  # Optional: only needed for LLM-based models
 ### Database Setup (Optional)
To run a production-ready PostgreSQL database instead of the default SQLite:

1. Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.
2. Start the database:
   ```bash
   docker-compose up -d
    ```

## Scraping 

This step collects raw data from GitHub, including repositories, organizations, contributor activity, and extended metadata (e.g., README, license, templates). All data is stored in a structured SQLite database located in:
```
Data/db/repository_data_{ACRONYM}_database.db
```

There are already configuration files available for **32 institutions** in `config/`:

- BIOHUB (Biohub)
- CMU (Carnegie Mellon University)
- ETH (ETH Zürich)
- GEORGIA_TECH (Georgia Institute of Technology)
- GWU (George Washington University)
- JHU (Johns Hopkins University)
- LERO (Lero — The Science Foundation Ireland Research Centre for Software)
- MGB (Mass General Brigham)
- MSU (Mississippi State University)
- OSU (Oregon State University)
- RIT (Rochester Institute of Technology)
- SLU (Saint Louis University)
- SNT (SnT – Université du Luxembourg)
- STANFORD (Stanford University)
- SYRACUSE (Syracuse University)
- TCD (Trinity College Dublin)
- UCB (University of California, Berkeley)
- UCD (University of California, Davis)
- UCI (University of California, Irvine)
- UCL (University College London)
- UCLA (University of California, Los Angeles)
- UCM (University of California, Merced)
- UCR (University of California, Riverside)
- UCSB (University of California, Santa Barbara)
- UCSC (University of California, Santa Cruz)
- UCSD (University of California, San Diego)
- UCSF (University of California, San Francisco)
- UGA (Université Grenoble Alpes)
- UMICH (University of Michigan)
- UT_AUSTIN (University of Texas at Austin)
- UVM (University of Vermont)
- UWMADISON (University of Wisconsin, Madison)

For a simple test case, replace `university_acronyms = ['UCSD']` in `repofinder/main_scraping.py` with the acronym of the institution you would like to collect data from (use the config slug, e.g. `georgia_tech`, `ut_austin`).

For any additional institution, create a configuration file inside the `config/` folder (e.g. `config/config_{acronym}.json`) and update the path accordingly.


Run the scraping script:
```sh
python repofinder/main_scraping.py
```


It will execute the following steps:
1. **Repository Finder:** Generates a JSON file with repositories based on a configuration file.
2. **Database Creation:** Reads the JSON file and creates a database.
3. **Organization Data Collection:** Gathers organization metadata.
4. **Repository Collection from Organizations:** Finds repositories owned by discovered organizations.
5. **User Data Collection:** Identifies users affiliated with the university.
6. **Repository Collection from Users:** Finds repositories owned by discovered users.
7. **Extra Features Extraction:** Retrieves extra features that are not collected by default (includes release downloads, readme, code of conduct, contributing, security policy, issue templates, pull request template, subscribers count).
8. **Contributor Data Collection:** Fetches contributor details for repositories.

Note: Execution times vary based on the number of repositories and API rate limits. You can selectively comment out steps in `main_scraping.py` to run only specific parts of the pipeline.

## Filtering
This step filters repositories in two stages: (1) identifying whether a repository is affiliated with a university and (2) classifying the type of project (DEV, EDU, DATA, DOCS, WEB, OTHER). Both tasks are handled in the `main_filtering.py` script.

1. Score-based classification, which applies a set of heuristic rules over repository and contributor metadata.
2. Supervised machine learning models using embedding models.
3. Large Language Model (LLM) classification using OpenAI models (e.g., GPT-4o, GPT-5-mini, and GPT-3.5-turbo).

There are manual labels and test sets for UCSB, UCSC and UCSD so you can use all classification methods for these three universities.

If you want to classify repositories for another university, provide a file named `{ACRONYM}_Random200.csv` with the columns `html_url` and `manual_label` in the following directory: 
     ```
  Data/manual_labels/{ACRONYM}_Random200.csv
     ```

Additionally, for ROC curve generation, you will need to provide a test set under:
     ```
  Data/test_data/test_set_{ACRONYM}.csv
     ```

For the type classification pipeline, we only use language models so no manual labels are required. However, to compute the accuracy of the classification, you will need to provide a test set under:
     ```
  Data/test_data/type_test_set_{ACRONYM}.csv
     ```

Project type test sets are provided for UCSB, UCSC and UCSD. 

Run the filtering script:
```sh
python repofinder/main_filtering.py
```

You can selectively comment out models in `main_filtering.py` to run only specific methods. This script generates prediction CSV files for each method (score-based, machine learning, and LLMs) in the `results/{ACRONYM}/` folder.

## Analysis
This step generates visual summaries and evaluation metrics based on the filtered repository data per university. It includes:

1. Language distribution 
2. License usage patterns
3. Adoption of best-practice repository features (e.g., README, security policy, issue templates)


Run the analysis script:
```sh
python repofinder/main_analysis.py
```

All plots are saved in the `plots/combined/` directory.

## Combined Type Analysis
This step generates visual summaries and evaluation metrics using filtered aggregated repository data. The script focuses on analyzing project characteristics by type and popularity.

It includes:

1. **Language distribution by project type**
2. **License usage patterns by project type**
3. **Adoption of best-practice repository features** (e.g., README, license, citation, issue templates)
4. **Project type distribution** across all affiliated repositories
5. **Feature heatmap by star buckets for DEV projects** — shows the presence of best practices in projects grouped by their GitHub star count
6. **Scatterplot of feature presence by stars** — visualizes how feature usage varies across individual projects based on their popularity

Run the combined analysis script:
```sh
python repofinder/main_analysis_combined.py
```

All plots are saved in the `plots/combined/` directory.










