from langchain_google_genai import GoogleGenerativeAI
from langchain_core.chat_sessions import ChatSession
import os

from langchain_core.prompts import ChatPromptTemplate

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import PromptTemplate

import pandas as pd
import numpy as np

import re
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta 

import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model 
from tensorflow.keras import losses
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.path.append('./libraries')
from database import get_results

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

llm = GoogleGenerativeAI(model="gemini-2.0-flash")
model = load_model('model_stmp.h5', custom_objects={'mse': losses.MeanSquaredError()})
encoders = joblib.load('encoders.pkl')
scaler = encoders['scaler']

prompt = PromptTemplate.from_template('''
{context}

Given the user question below, classify it as either being about `Prediction`, `Insights`, or `General`. 
- If the question follows the context given above, classify it in either of the topics above. \
    Take in mind that for prediction should be in future data and not past data, and for Insights is based on the previous data.
- If the question does not match the context or is outside of the context, classify it as `General`.

Respond with ONLY one word: the classification.

Question: 
{question}

Classification:''')

chain = (prompt | llm | StrOutputParser())

context = '''
Act as an inventory manager for Global Trends Apparel (GTA), an international \
fashion retailer operating in 10 countries. Your role is to manage inventory across \
200 stores and warehouses, ensuring stock levels are optimized to avoid \
stockouts and overstocking. You must consider regional demand, seasonality, and cultural factors.

Business Context: GTA offers a wide range of products, including casual wear, activewear, and seasonal fashion. \
The company operates in various regions with different climates and shopping patterns. \
Efficient inventory management is crucial for meeting customer demand and minimizing costs.

Challenges:

    - Stockouts in high-demand regions.
    - Overstocking in low-demand regions.
    - Difficulty forecasting demand due to changing seasons and local events.
    - Inventory movement between stores across regions.
    - Delayed replenishment of popular items.

Objective:

    - Forecast demand based on regional factors.
    - Optimize stock levels to prevent stockouts and overstocking.
    - Recommend inventory movements between stores.
    - Provide alerts for timely replenishment.
'''

schemas = '''
CREATE TABLE products (
    gtin            CHAR(13),
    product_code    VARCHAR(15),
    size            VARCHAR(10),
    color           VARCHAR(30),
    label           VARCHAR(100),
    category        VARCHAR(50),

    PRIMARY KEY (gtin, product_code)
);

CREATE TABLE sales (
    sku         VARCHAR(15),
    quantity    SMALLINT,
    site_code   VARCHAR(8),
    date        DATE
);

CREATE TABLE soh (
    site_code   VARCHAR(8),
    sku         VARCHAR(15),
    quantity    INTEGER,
    date        DATE, 

    PRIMARY KEY (site_code, sku, date) -- composed pk
);
CREATE INDEX idx_sales_sku_site_code ON sales (sku, site_code);
CREATE INDEX idx_soh_site_sku_date ON soh (site_code, sku, date);
CREATE INDEX idx_products_category ON products (category);

CREATE MATERIALIZED VIEW soh_batch_3_months as 
SELECT 
    sku,
    site_code,
    DATE_TRUNC('quarter', date) AS batch_date,
    quantity
FROM (
    SELECT 
        sku,
        site_code,
        date,
        quantity,
        ROW_NUMBER() OVER (
            PARTITION BY sku, site_code, DATE_TRUNC('quarter', date)
            ORDER BY date DESC
        ) AS row_number
    FROM soh
) subquery
WHERE row_number = 1;

-- take in mind that the soh table stores only the information of actual stock based on the date, sku and site_code
'''

general_chain = (
    PromptTemplate.from_template('''
Context:
{context}

Instructions:
1.  Begin with a polite greeting.
2.  Clearly and concisely introduce yourself and your purpose, drawing directly from the provided context.
3.  State the specific tasks or questions you are designed to handle.
4.  Maintain a professional and helpful tone.
5.  Keep the introduction brief and to the point.
6.  Do not invent functionalities that are not described in the context.
7.  Output ONLY the introduction. Do not include any additional explanations or markdown formatting.
    ''')
    | llm
    | StrOutputParser()
)

forecast_chain = (
    PromptTemplate.from_template('''
{context}

Please carefully analyze the provided question and extract the following information in the JSON format:

- **start_date**: The start date in the format **%Y-%m-%d**. If not specified, then today.
- **end_date**: The end date in the format **%Y-%m-%d**. If not specified, then today.
- **categories**: A list of categories, which must be one or more of the following values:  
  `['Outerwear', 'Tops', 'Bottoms', 'Dresses', 'Swimwear', 'Activewear', 'Footwear', 'Accessories']`. 
  If no categories are specified, try to extract them from the products mentioned in the question. \
  Return all categories by default if no categories can be identified from the products. \
  Always you need to output the categories from the list above following the format for each category
- **countries**: A list of countries where the company has a presence. The list of possible countries is:  
  `['USA', 'Canada', 'UK', 'France', 'Germany', 'Australia', 'India', 'Japan', 'Brazil', 'Mexico']`.

Please ensure the following rules are followed:

1. **Date Format**: Both the **start_date** and **end_date** MUST be in the format: **%Y-%m-%d**.
2. **Categories**: If the question doesn't specify categories, attempt to infer them from the products mentioned. \
    If that isn't possible, include all categories by default.
3. **Countries**: Only include countries from the list of valid countries where the company is present.

Question:  
{question}

Result:
    ''')
    | llm
)

output_predictions_chain = (
    PromptTemplate.from_template('''
**Context**
{context}

Given the predicted sales for each store, product category, in a range of dates, \
along with the actual current inventory levels in each store, \
your task is to efficiently manage the stock in each store. \
The goal is to optimize inventory levels to minimize both stockouts and overstocking, \
ensuring that each store has the right quantity of products at the right time.

**Predictions of Sales**  
{data}

**Actual Inventory Levels:**  
{inventory}

**Instructions**  
Using the provided predicted sales data and current inventory levels, \
formulate a strategy for managing the inventory. \
Your objective is to adjust the stock levels in each store such that:

1. **Stockouts** are minimized: Ensure that stores never run out of stock for high-demand products, \
    especially those with high sales forecasts.
2. **Overstocking** is avoided: Prevent excessive inventory accumulation in stores, particularly \
    for slow-moving products while maintaining optimal stock levels.
3. **Efficient Distribution**: Distribute products across stores in such a way \
    that demand in all locations is met without unnecessary overstocking.
4. **Prioritize critical products**: For high-priority categories or high-demand products, \
    consider reallocation or restocking strategies to avoid shortages.

Please consider factors such as the predicted demand, available stock, and the \
store's capacity to hold inventory. Provide a clear and actionable inventory plan \
based only on the data provided before, you only have to output that as brief as possible.

**Response:**
    ''')
    | llm
    | StrOutputParser()
)

def get_site_codes(): 
    return get_results('''
SELECT 
    DISTINCT site_code
FROM 
    sales
    ''').site_code.values
def segregate_site_codes(countries):
    site_codes = get_site_codes()
    result = []
    for site_code in site_codes:
        for country in countries:
            if country.upper().startswith(re.match(r'([A-Za-z]+)\d+', site_code).group(1)):
                result.append(site_code)
    return result

def load_data(skus, site_codes):
    """Loads the data from cloud storage"""
    formatted_skus = ', '.join(f"'{sku}'" for sku in skus)

    formatted_site_codes = ', '.join(f"'{site_code}'" for site_code in site_codes)
    query = f"""
SELECT 
    s.sku, 
    SUM(s.quantity) as quantity, 
    s.date, 
    s.site_code, 
    p.category
FROM 
    sales s
LEFT JOIN 
    products p on p.product_code = s.sku
WHERE 
    s.sku IN ({formatted_skus}) AND s.site_code IN ({formatted_site_codes})
GROUP BY 
    s.date, s.sku, p.category, s.site_code
ORDER BY 
    s.site_code, s.sku, s.date
    """
    return get_results(query)

def preprocess_data(df_batch, encoders):
    df_batch['date'] = pd.to_datetime(df_batch['date'])
    df_batch['day_of_week'] = df_batch['date'].dt.dayofweek
    df_batch['month'] = df_batch['date'].dt.month

    df_batch['site_code'] = encoders['site_code'].transform(df_batch['site_code'])
    df_batch['sku'] = encoders['sku'].transform(df_batch['sku'])
    df_batch['category'] = encoders['category'].transform(df_batch['category'])
    df_batch['season'] = df_batch['date'].apply(lambda x: (x.month - 1) // 3)
    df_batch['season'] = encoders['season'].transform(df_batch['season'])

    numerical_features = ['quantity', 'day_of_week', 'month']
    scaler = StandardScaler()
    df_batch[numerical_features] = scaler.fit_transform(df_batch[numerical_features])

    return df_batch, scaler

def create_batch(df_batch, sequence_length):
    numerical_data = df_batch[['quantity', 'day_of_week', 'month']].values
    site_code_data = df_batch['site_code'].values
    sku_data = df_batch['sku'].values
    category_data = df_batch['category'].values
    season_data = df_batch['season'].values
    target_data = df_batch['quantity'].values

    numerical_batches = []
    site_code_batches = []
    sku_batches = []
    category_batches = []
    season_batches = []
    target_batches = []

    for i in range(0, len(df_batch) - sequence_length):
        numerical_batches.append(numerical_data[i:i + sequence_length])
        site_code_batches.append(site_code_data[i:i + sequence_length])
        sku_batches.append(sku_data[i:i + sequence_length])
        category_batches.append(category_data[i:i + sequence_length])
        season_batches.append(season_data[i:i + sequence_length])
        target_batches.append(target_data[i + sequence_length])

    numerical_batches = pad_sequences(numerical_batches, dtype='float32')
    site_code_batches = pad_sequences(site_code_batches, dtype='int32')
    sku_batches = pad_sequences(sku_batches, dtype='int32')
    category_batches = pad_sequences(category_batches, dtype='int32')
    season_batches = pad_sequences(season_batches, dtype='int32')
    target_batches = np.array(target_batches)

    return numerical_batches, site_code_batches, sku_batches, category_batches, season_batches, target_batches


def predict(sku, site_code, sequence_length=10, prediction_dates=None):
    """Makes predictions using the trained model for specified dates."""

    skus = [sku]
    site_codes = [site_code]
    df_predict = load_data(skus, site_codes)

    # Filter by dates if provided
    if prediction_dates is not None:
        #df_predict['date'] = pd.to_datetime(df_predict['date'])

        if df_predict.empty: return None

    df_predict, _ = preprocess_data(df_predict, encoders)
    numerical_batches, site_code_batches, sku_batches, category_batches, season_batches, _ = create_batch(df_predict, sequence_length)

    predictions = model.predict([numerical_batches, site_code_batches, sku_batches, category_batches, season_batches], verbose=1)

    original_scale_predictions = scaler.inverse_transform(np.concatenate((np.zeros_like(predictions), np.zeros_like(predictions), predictions), axis=1))[:, 2]

    return original_scale_predictions

def prediction_branch(x):
    try: 
        
        data = forecast_chain.invoke(x)
        data =  json.loads(re.sub(r'```json\n|\n```','',data))

        data['site_codes'] = segregate_site_codes(data['countries'])
        
        formatted_site_codes = ', '.join(f"'{site_code}'" for site_code in data['site_codes'])

        inventory = get_results(f'''
SELECT 
    s.sku,
    p.category, 
    SUM(s.quantity) as quantity, 
    s.date, 
    p.label
FROM
    soh s 
INNER JOIN 
    products p ON p.product_code = s.sku
WHERE s.site_code in ({formatted_site_codes})
    AND date in (
            SELECT MAX(date) 
            FROM soh 
        ) 
GROUP BY p.category, s.sku, p.label, s.date
LIMIT 3;
        ''')
        skus = inventory.sku.unique().tolist()
        labels = inventory.label.unique().tolist()

        prediction_dates = pd.to_datetime([data['start_date'], data['end_date']])
        prediction_data = []
        for i in range(len(skus)):
            sku, label = skus[i], labels[i]
            for site_code in data['site_codes']:
                predictions = predict(sku, site_code, prediction_dates=prediction_dates)
                if predictions is None: predictions = [0]
                prediction_data.append({
                    'sku': sku, 
                    'site_code': site_code,
                    'predicted_sales': sum(predictions),
                    'label': label
                })
        predicted_df = pd.DataFrame(prediction_data)
        return output_predictions_chain.invoke({
            'context': x['context'], 
            'data': predicted_df,
            'inventory': inventory,
        })
    except Exception as e:
        print(e)
        return 'An error occurred, please try again later.'

insights_chain = (
    PromptTemplate.from_template('''
Context:
{context}

Objective:
Generate an optimized SQL query to answer the user's question, focusing on delivering data suitable for insightful visualizations.

Database Schema:
{schemas}

User Query:
{question}

Requirements:
1.  Craft a single SQL query that directly addresses the user's information request.
2.  Design the query with data visualization in mind. This includes:
    * Appropriate aggregations (SUM, AVG, COUNT, etc.).
    * Effective grouping (GROUP BY clauses).
    * Sorting (ORDER BY clauses) for clear presentation.
    * Use of aliases for readability.
3.  Ensure the query is efficient and adheres to SQL best practices.
4.  If the query involves date/time manipulations, handle these data types correctly.
5.  If the user's query suggests comparisons or trend analysis, reflect this in the SQL structure.
6.  Return ONLY the SQL query. No additional text, explanations, or markdown formatting is required.
7.  Strictly rely on the provided database schema. Do not invent or assume data that is not present.
8.  **Site Code Constraints:**
    * When filtering by `site_code`, consider the following country associations: `['USA', 'Canada', 'UK', 'France', 'Germany', 'Australia', 'India', 'Japan', 'Brazil', 'Mexico']`.
    * Site codes are structured as `[Country Code]XXX` (e.g., `USA001`, `CAN002`, etc), with the exception of `UK` codes, which are `UK000`, etc.
    * JAPAN SHOULD BE JAPXXX where X is a digit
9.  **Category Constraints:**
    * When filtering by `category`, use only the following categories, maintaining the exact case: `['Outerwear', 'Tops', 'Bottoms', 'Dresses', 'Swimwear', 'Activewear', 'Footwear', 'Accessories']`.
10. **Product Code Exclusion:**
    * Do not generate or query based on product codes or any identifiers not explicitly defined in the provided schema.
11. **Label Exclusion:**
    * Only use the parameter label in clauses such: GROUP BY, ORDER BY, SELECT
12. **SOH Quantity Handling:**
    * **Crucially, do not attempt to SUM the `soh.quantity` column directly.** 
    The `soh.quantity` represents the remaining stock at the end of each day. 
    Summing this column will produce misleading results. Instead, focus on retrieving the latest `soh.quantity` \
    values for the requested criteria.

Output:
    ''')
    | llm 
)

output_insights_chain = (
    PromptTemplate.from_template('''
Task:
You are an expert data formatter. Your task is to take the provided Pandas DataFrame and present it in a clear, \
concise, and user-friendly format. The goal is to make the data easily digestible and understandable for a non-technical user. \
Based on the question of the user: 

Question: 
{question}

Input:
{data}

Instructions:
1.  **Readability:** Prioritize readability. Use clear and descriptive column names. \
If necessary, rephrase column names to be more user-friendly.
2.  **Conciseness:** Avoid overwhelming the user with unnecessary details. Focus on the most important information.
3.  **Context:** Provide a brief introductory sentence or two explaining the data being presented.
4.  **Formatting:** Use appropriate formatting techniques to enhance clarity, such as:
    * Rounding numerical values to a reasonable number of decimal places.
    * Formatting dates into a user-friendly format (e.g., "YYYY-MM-DD").
    * Using commas or other separators for large numbers.
5.  **Summarization:** If the DataFrame is large, provide a summary of key findings or trends.
6.  **Avoid Technical Jargon:** Use plain language that is easily understandable by a non-technical audience.
7.  **Table Presentation:** if it is a table, present it as a table using markdown format.
8.  **List Presentation:** If it is a list, present it as a clear list.
9.  **Do not invent data:** strictly use the data provided.
10. **Do not perform any calculations or data analysis:** only format the data.
11. **Output ONLY the formatted data and the intro text**. \
Do not include any additional explanations or markdown formatting outside of the table or list.

Output:
    ''')
    | llm 
    | StrOutputParser()
)

def insights_branch(x):
    sql = insights_chain.invoke(x)
    sql = re.sub(r'```sql\n|\n```','',sql)
    print(sql)
    data = get_results(sql)
    return output_insights_chain.invoke({
        'data': data,
        'question': x['question']
    })

decision_tree = RunnableBranch(
    (lambda x: 'prediction' in x['topic'].lower(), prediction_branch),
    (lambda x: 'insights' in x['topic'].lower(), insights_branch),
    general_chain
)

full_chain = (
    {
        'topic': chain, 
        'question': lambda x: x['question'], 
        'context': lambda x: x['context'],
        'schemas': lambda x: x['schemas'],
    } 
    | decision_tree)

def get_response(question):
    return full_chain.invoke({
        'question': question,
        'context': context,
        'schemas': schemas,
    })
