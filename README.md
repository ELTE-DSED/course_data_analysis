# Analysis of University Cybersecurity Course Descriptions using NLP

This project uses Natural Language Processing (NLP) techniques to analyze a dataset of university cybersecurity course descriptions. The primary goal is to process, understand, and extract insights from these descriptions, including identifying key topics, extracting relevant skills, and measuring similarity between courses.

## Motivation

Technology universities offer numerous cybersecurity courses, but their descriptions often lack consistency and clarity. This project aims to:

* Uncover hidden patterns and key themes in course descriptions.
* Extract cybersecurity skills mentioned in the courses.
* Analyze the similarity between different courses based on their content and skills.
* Lay the groundwork for comparing university offerings with job market demands.

## Dataset

The analysis is performed on an Excel file named `dataset.xlsx`. Each row represents a university course and includes columns such as:

* `university_name`
* `study_program_name`
* `city`
* `country`
* `description` (The primary text field for our analysis)

**Note:** This notebook currently loads the dataset from a specific path on Google Drive (`/content/drive/MyDrive/dslab/dataset.xlsx`). 
The dataset is uploaded in the repository as `dataset.xlxs`, so it can be downloaded from here.

## Key Features & Techniques Used

This notebook demonstrates a variety of NLP and data analysis techniques:

* **Data Exploration:** Using Pandas for loading and initial inspection, Matplotlib and Seaborn for visualizing text length distributions.
* **Language Detection:** Using `langdetect` to ensure descriptions are primarily in English.
* **Text Preprocessing:**
    * Lowercasing, punctuation removal (`re`).
    * Tokenization (`nltk.word_tokenize`).
    * Stopword removal (`nltk.corpus.stopwords`).
* **Frequency Analysis:**
    * Identifying most frequent words (`collections.Counter`).
    * Visualizing frequencies with bar plots and Word Clouds (`matplotlib`, `seaborn`, `wordcloud`).
    * N-gram (Bigram & Trigram) analysis (`nltk.util.ngrams`).
* **Vectorization & Topic Modeling:**
    * TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`).
    * Latent Dirichlet Allocation (LDA) for topic discovery (`sklearn.decomposition.LatentDirichletAllocation`).
* **Named Entity Recognition (NER):** Using `spaCy` to identify organizations, technologies, etc.
* **Graph Analytics:** Building and visualizing entity relationships with `networkx`.
* **Skill Extraction:**
    * Using `spaCy` for noun phrase extraction.
    * Using `fuzzywuzzy` for approximate matching against a predefined skill list.
* **Similarity Analysis:**
    * Jaccard Similarity based on extracted skills.
    * Cosine Similarity using TF-IDF vectors.
    * Semantic Similarity using `sentence-transformers` embeddings.
* **Experimental Embeddings:** Setup for using `RETVec`.

## Setup & Installation

This project was developed in Google Colab. To run it locally or in another environment, Python 3 and the following libraries are needed. They can be can install using following pip command:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn spacy langdetect fuzzywuzzy[speedup] sentence-transformers torch torchvision retvec tensorflow tensorflow-text openpyxl
```

*Note: openpyxl is needed by Pandas to read .xlsx files.*

The necessary data for NLTK and spaCy are also have to be downloaded:

```python
import nltk
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

spacy.cli.download('en_core_web_sm')
```

**Note:** The notebook uses a RETVec model saved and loaded from the specified path (/content/drive/MyDrive/dslab/retvec_model).
To be able to run it locally, the `saved_model.pb` from the reposiroty has to be downloaded and used in the notebook.

## How to Use
Clone the repository:

```bash
git clone course_data_analysis
cd course_data_analysis
```

* **Install dependencies**: Follow the steps in the "Setup & Installation" section.
* **Ensure Dataset Availability**: Make sure the dataset.xlsx file is accessible and update the pd.read_excel path in the notebook if necessary.
* **Run the Notebook**: Open dslab1.ipynb in a Jupyter environment (like Jupyter Lab, Jupyter Notebook, or VS Code with Python/Jupyter extensions) or upload it to Google Colab.
* **Execute the cells**: Run the cells sequentially to perform the analysis.


## Future Work
* Refine the skill extraction process with a more extensive skill list or advanced extraction methods.
* Utilize more advanced text embedding models (like BERT or domain-specific models) for improved similarity and clustering.
* Compare the extracted skills and topics with actual cybersecurity job market requirements.
* Develop an interactive dashboard to explore the results.
