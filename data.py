"""
Instructions for Implementing data.py

Tabular Data Functions
Loading: Define a function to load tabular data from the provided formats and return it in a structured format of your choice.

Image Data Functions
Loading Images: Define a function to load all images from the specified folder.
Preprocessing: Create a function to preprocess images, such as normalizing pixel values or converting image formats if necessary.

Text Data Functions
Loading: Define a function to load text data from the provided file.
Preprocessing: Create a function to preprocess text data, including operations such as converting to lowercase, removing unwanted characters, and one-hot encoding the text.

Data Splitting
Develop a generic function to split data into training, validation, and test sets. Ensure the function accepts data and corresponding labels, along with parameters for split ratios and reproducibility.

Storing/Saving Data
Create a function to save processed data to disk. The function should handle different data types and file formats based on the data provided.
"""
import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

def load_tabular_data(input_path:str, output_format: str = 'pandas'):
    """
    input_path: string
        path to the input file (csv/excel/parquet).
    output_format: string
        choice of the ouput structured format (pandas/numpy)
    (the function currently cannot handle tabular data that contains non-numerical elements.)
    """

    # detect input file type and read
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    elif input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        raise ValueError('Unsupported file format input')

    # transform df to the structured format of user's choice
    if output_format == 'pandas':
        return df
    elif output_format == 'numpy':
        df_np = df.to_numpy()
        return df_np
    else:
        raise ValueError("Output format must be one of ['pandas','numpy','torch']")

def load_image_data(folder_path: str) -> list:
    """
    load all images from a specified folder.
    (make sure the folder only contains image files)
    """

    images = []
    
    # iterate through the folder to read images and append them to a list
    for image_name in os.listdir(folder_path):
        path = os.path.join(folder_path, image_name)
        image = Image.open(path).convert('RGB')
        images.append(image)
    return images

def preprocess_image_data(images:list, target_size=(224,224)) -> np.ndarray:
    """
    preprocess a list of images for it to be ready for model trainning.
        resize -> convert -> normalize -> output(numpy)
    """

    processed = []

    for image in images:

        # resize and convert to np array
        image_resized = image.resize(target_size)
        image_arr = np.array(image_resized, dtype='float32')
        # normalize
        image_arr = image_arr / 255.0

        processed.append(image_arr)
      
    return processed

def load_text_data(input_path:str) -> list:
    """
    load text data from given path, consider the given file type(txt/csv)
    for csv input, the file should only contain one column.
    output the result as a list that contains strings(line by line).
    """
    text = []

    if input_path.endswith('.csv'):

        df = pd.read_csv(input_path)
   
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must contain 'review' and 'sentiment' columns")
 
        reviews = df['review'].astype(str).apply(
            lambda x: re.sub(r'<.*?>', '', x)  
        ).tolist()
   
        labels = df['sentiment'].apply(
            lambda x: 1 if str(x).lower() == 'positive' else 0
        ).tolist()
        
        return reviews, labels
    
    elif input_path.endswith('.txt'):
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    text.append(line)
        return text
    else:
        raise ValueError('Unsupported file format input')

def load_numerical_text(input_path: str):
    """
    Load numerical txt file (space-separated values) into numpy arrays.
    """
    lines = load_text_data(input_path)  
    data = [list(map(float, line.split())) for line in lines]
    data = np.array(data)
    return data


def preprocess_text_data(text:list, vectorize_method: str = 'one-hot', remove_stop_words:bool = False, output_format: str = "numpy"):
    """
    This function aims at completing two following things:
    1. do basic cleaning jobs like converting to lowercase and removing unwanted characters.
    2. vectorize the cleaned text list for further model training use.
    3. can output different types of data based on users' choices(numpy/sparse).
    """

    cleaned = []
    for t in text:
        t_cleaned = t.lower()
        t_cleaned = re.sub(r"[^\w\s]", "", t_cleaned)
        t_cleaned = re.sub(r"\s+", "", t_cleaned).strip()
        cleaned.append(t_cleaned)
    
    if remove_stop_words:
        stop_opt = 'english'
    else:
        stop_opt = None

    if vectorize_method == 'one-hot':
        vec = CountVectorizer(binary=True, stop_words=stop_opt)
    elif vectorize_method == 'count':
        vec = CountVectorizer(stop_words=stop_opt)
    elif vectorize_method == 'tfidf':
        vec = TfidfVectorizer(stop_words=stop_opt)
    else:
        raise ValueError("vectorize_method must be one of ['one-hot','count','tfidf']")
    
    x = vec.fit_transform(cleaned)
    
    if output_format == 'numpy':
        return x.toarray(), vec
    elif output_format == 'sparse':
        return x, vec
    else: 
        raise ValueError('Unsuppiorted output format.')

def split_data(features, labels, val_ratio: float =0.1, test_ratio:float=0.2, random_state: int = 6600):
    """
    split features and labels into train/val/test sets.
    ratios are all based on the whole number of data.
    """

    # split test dataset 
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_ratio, random_state=random_state)

    # split validation dataset
    val_ratio_relative = val_ratio / (1 - test_ratio)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio_relative, random_state=random_state)

    return x_train, x_val, x_test, y_train, y_val, y_test


def save_data(obj, save_path: str):
    """
    Save data based on type: np.ndarray / pandas.DataFrame / scipy.sparse
    """

    if isinstance(obj, np.ndarray):
        if not save_path.endswith(".npy"):
            save_path = save_path + ".npy"
        np.save(save_path, obj)
        print(f"The object has been saved to {save_path}")

    elif isinstance(obj, pd.DataFrame):
        if save_path.endswith(".parquet"):
            obj.to_parquet(save_path, index=False)
        elif save_path.endswith(".csv"):
            obj.to_csv(save_path, index=False)
        else:
            raise ValueError("For DataFrame, please use .csv or .parquet extension")
        print(f"The object has been saved to {save_path}")

    elif sp.issparse(obj):
        if not save_path.endswith(".npz"):
            save_path = save_path + ".npz"
        sp.save_npz(save_path, obj)
        print(f"The object has been saved to {save_path}")

    else:
        raise ValueError("Unsupported data type, expected numpy/pandas/sparse.")









    



