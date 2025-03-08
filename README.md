# Repository Finder

## Overview
This tool identifies and analyzes open-source repositories affiliated with universities using GitHub metadata and contributor analysis. Running `main.py` will execute the full pipeline, fetching repository data, processing metadata, and storing it in a database.

## Features
- **Repository Discovery:** Extracts repositories based on a configuration file.
- **Database Population:** Creates and fills a database with repository information.
- **Organization Analysis:** Gathers organization-related data.
- **README Extraction:** Retrieves README content from repositories.
- **Contributor Analysis:** Collects contributor details and statistics.

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
     GITHUB_API_TOKEN=your_personal_access_token
     ```

## Usage
There are already configuration files available for six universities:
- UCSC (University of California, Santa Cruz)
- UCSD (University of California, San Diego)
- UCLA (University of California, Los Angeles)
- UCB (University of California, Berkeley)
- UCSB (University of California, Santa Barbara)
- UCD (University of California, Davis)

For a simple test case, replace `university_acronyms = ['UCSD']` in `main.py` with the acronym of the university you would like to collect data from.

For any other university, create a configuration file inside the config folder and update the path accordingly.


Run the main script:
```sh
python main.py
```
This will execute the following steps:
1. **Repository Finder:** Generates a JSON file with repositories based on a configuration file (~5 mins).
2. **Database Creation:** Reads the JSON file and creates a database (~1 secs).
3. **Organization Data Collection:** Gathers organization metadata (~1 hour).
4. **README Extraction:** Retrieves README content (~1 hour).
5. **Contributor Data Collection:** Fetches contributor details (~4 hours).

Execution times may vary based on the number of repositories and API rate limits.

