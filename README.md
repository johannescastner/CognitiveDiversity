# Diversity Measurement with SFR-Embedding-Mistral

This repository contains a Python script that utilizes the SFR-Embedding-Mistral model from Salesforce to calculate the diversity of opinions in a set of text statements. The script generates embeddings for each statement and applies a diversity measurement based on the Jensen-Shannon Divergence.

## Prerequisites

Before running the script, ensure you have the following installed:
- Python 3.6 or higher
- PyTorch
- Hugging Face's Transformers library

You can install the required libraries using pip:

`pip install torch transformers`


## Usage

1. Clone this repository to your local machine using:

`git clone https://github.com/johannescastner/CognitiveDiversity.git`


2. Navigate to the repository directory:

`cd your-repository-name`


3. Run the script:

`python diversity_measurement.py`


The script will output the "Diversity of opinions" value based on the predefined statements within the script.

## Customizing the Statements

To customize the statements, modify the `statements` list in the `diversity_measurement.py` script. Replace the existing statements with your own, ensuring each statement reflects a unique opinion or perspective.

## Understanding the Output

The output "Diversity of opinions" is a numerical value that represents the diversity of the provided statements based on their semantic content. Higher values indicate greater diversity.

## Contributing

Contributions to this project are welcome! Feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
