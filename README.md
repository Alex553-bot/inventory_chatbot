# Inventory Management for Retail Clothing Company

This project is designed to manage the inventory of a retail clothing company using synthetic data. It includes a chatbot that interacts with the inventory data and performs various tasks related to stock management. The system is built using a variety of Python libraries and frameworks, and it aims to improve inventory tracking and customer service in a retail environment.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)

## Project Overview

The project aims to streamline inventory management in a retail clothing business by:

- Simulating inventory data (e.g., products, categories, stock levels).
- Doing the ETL process using batch and streaming.
- Storing to a cloud storage
- Using machine learning models and frameworks to generate synthetic data for realistic simulations.
- Providing a chatbot interface using `Streamlit` that interacts with the inventory and assists employees in managing stock levels, placing orders, and retrieving product details.

By integrating AI-driven automation with real-time data, this system will enhance decision-making and improve operational efficiency in the retail store.

## Features

- **Inventory Tracking**: Track stock levels, product details, and categories in real time.
- **Data Generation**: Use synthetic data for simulating realistic inventory scenarios.
- **Chatbot Interface**: A conversational agent for interacting with inventory data. Employees can check stock, place orders, and query product details.
- **Product Categorization**: Automatically classify products into categories such as "Tops", "Bottoms", "Accessories", etc.
- **Predict Stocks**: Notify users when stock levels fall below a predefined threshold and/or in a near future.

## Tech Stack

- **Programming Language**: Python
- **Libraries/Frameworks**:
  - `pandas`: Data manipulation and analysis.
  - `numpy`: Numerical computations.
  - `scikit-learn`: Machine learning for reshape data.
  - `TensorFlow`: Create a LSTM model.
  - `psycopg2`: Database ORM for interacting with the database.
  - `streamlit`: Create a simple but powerful user interface.
  - `langchain`: NLP process.
  - `faker`: generate synthetic data.
  - `confluent-kafka`: communication with Kafka cluster.
  - `pyspark`: control high volumes of data.
  - `Git`: Version control.

## Setup and Installation

Follow the steps below to set up the project on your local machine:

### 1. Clone the Repository

```bash
git clone git@github.com:Alex553-bot/inventory_chatbot.git
cd inventory_chatbot
```
### 2. Install Required Libraries

Install the necessary dependencies:

```bash
pip install -r --break-system-packages requirements.txt
```

## Usage 

Take in mind that you have to generate the data and initialize the kafka cluster, also the database.

To run the chatbot:

```bash
streamlit run chatbot.py
```

## Project Structure

```bash
inventory-chatbot/
├── chatbot.py
├── requirements.txt
├── model_stmp.h5
├── module.py 
├── encoders.pkl
├── data/
    ├── distribution_of_sales_by_country.csv
    ├── distribution_of_sales_by_season.csv
    ├── products.csv
    ├── sales.csv
    ├── soh.csv
├── libraries/
    ├── database.py
    ├── utils.py
├── notebooks/
    ├── simulate_data/
        ├── generating-batch-products.ipynb
        ├── generating_batch_sales.ipynb
        ├── generating_batch_soh.ipynb
        ├── generating_client_behaviour.ipynb
        ├── generating_stream_data.ipynb
    ├── business_intelligence/
        ├── data_analysis.ipynb
    ├── ETL/
        ├── batch_pyspark.ipynb
        ├── consumer_streaming_sales.ipynb
        ├── consumer_streaming_soh.ipynb
        ├── main.sql
    ├── service/
        ├── langchain.ipynb
        ├── mode_train.ipynb
```