# RAPTOR RAG

## Overview

This project focuses on training a machine learning model to recognize patterns in data using the Raptor model and storing the resulting embeddings in a vector database. The main components of the project include data preprocessing, model training, and embedding management.

## Project Structure

- **raptor-training.ipynb**: This Jupyter notebook contains the workflow for training the Raptor model on a dataset. It includes data loading, preprocessing, model configuration, and evaluation metrics.
  
- **vector_db.ipynb**: This notebook demonstrates how to create, manage, and query a vector database. It integrates with the embeddings generated from the Raptor model to enable efficient similarity searches.

- **app.py**: A Python script that serves as the application interface, allowing users to interact with the trained model and vector database. It implements a simple web application to facilitate user queries and display results.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/raptor-training-vector-db.git
   cd raptor-training-vector-db
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have the necessary data files in the specified directory before running the notebooks and the application.

## Usage

### 1. Training the Model

Open the `raptor-training.ipynb` notebook and execute the cells to train the Raptor model. This process involves:

- Loading the dataset.
- Preprocessing the data.
- store text summaries in joblib format

### 2. Managing the Vector Database

Once the model is trained, you can open the `vector_db.ipynb` notebook to:

- Generate embeddings from the trained model (Gemini Embeddings).
- Store the embeddings in a vector database (Milvus).
- Query the database for similar items.
- also for the vector db we set up a milvus docker image using following commands
 ```bash
   curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
   bash standalone_embed.sh start
   ```

### 3. Running the Application

Run the application by executing the following command:
```bash
python app.py
```
This will start a local web server, and you can access the application at `http://127.0.0.1:5000`.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to enhance the functionality or improve the documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Raptor Model](https://arxiv.org/pdf/2401.18059) - For the machine learning architecture.
- [Vector Database]([https://example.com](https://milvus.io/docs/install_standalone-docker.md)) - For enabling fast retrieval of embeddings.
