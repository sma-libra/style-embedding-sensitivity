#%pip install mistralai -q
#pip install openai -q
#pip install voyageai
#pip install openai==0.28

import os
import numpy as np
import pickle
import glob
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
import openai
import time
import voyageai
import google.generativeai as genai
import textwrap
import yaml

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)


mistral_client = Mistral(api_key=config['api_keys']['mistralai'])
openai.api_key = config['api_keys']['openai']
genai.configure(api_key=config['api_keys']['genai'])

##########################################################################
#  CHANGE HERE 
##########################################################################
# Define the texts paths for embedding extraction
all_paths = {

    # "proust_mistral_gen": "../Texts/STYLE_GEN/PROUST_MISTRAL_GEN", 
    # "yourcenar_mistral_gen": "../Texts/STYLE_GEN/YOURCENAR_MISTRAL_GEN", 
    # "celine_mistral_gen": "../Texts/STYLE_GEN/CELINE_MISTRAL_GEN", 

    # "proust_gpt_gen": "../Texts/STYLE_GEN/PROUST_GPT_GEN", 
    # "yourcenar_gpt_gen": "../Texts/STYLE_GEN/YOURCENAR_GPT_GEN", 
    # "celine_gpt_gen": "../Texts/STYLE_GEN/CELINE_GPT_GEN", 
    
    # "proust_gemini_gen": "../Texts/STYLE_GEN/PROUST_GEMINI_GEN", 
    # "yourcenar_gemini_gen": "../Texts/STYLE_GEN/YOURCENAR_GEMINI_GEN", 
    # "celine_gemini_gen": "../Texts/STYLE_GEN/CELINE_GEMINI_GEN", 
    
    # "tuffery_ref": "../Texts/TUFFERY_REF",  
    # "proust_ref": "../Texts/STYLE_REF/PROUST_REF", 
    # "yourcenar_ref": "../Texts/STYLE_REF/YOURCENAR_REF", 
    # "celine_ref": "../Texts/STYLE_REF/CELINE_REF", 

}


# Embedding generation functions
def get_text_embedding_mistral(input_text, model_path):
    embeddings_batch_response = mistral_client.embeddings.create(
        model=model_path,
        inputs=input_text
    )
    return np.array(embeddings_batch_response.data[0].embedding)

def get_text_embedding_openai(input_text, model_path):
    response = openai.Embedding.create(
        model=model_path,
        input=input_text
    )
    return np.array(response['data'][0]['embedding'])

def get_text_embedding_voyageai(input_text, model_path):
    response = voyageai.Client(api_key=config['api_keys']['voyageai']).embed(
        model=model_path,
        texts=[input_text]
    )
    return np.array(response.embeddings[0])

def get_text_embedding_genai(input_text, model_path):

    max_bytes = 9500
    """Truncates a string to fit within the max_bytes limit."""
    encoded_text = input_text.encode('utf-8')  # Convert text to bytes
    if len(encoded_text) <= max_bytes:
        truncated_text = input_text
    else:
    # Truncate at the last valid character boundary
        truncated_text = encoded_text[:max_bytes].decode('utf-8', errors='ignore')


    result = genai.embed_content(
        model=model_path,
        content=truncated_text)
    
    return np.array(result['embedding'])


##########################################################################
#  CHANGE HERE 
##########################################################################


# Model configurations
# Advice : run voyage alone if possible, because rate limits are lower
model_configs = [
    {"model_path": "mistral-embed", "model_size": 1024, "embedding_func": get_text_embedding_mistral},
    {"model_path": "text-embedding-3-small", "model_size": 1536, "embedding_func": get_text_embedding_openai},
    {"model_path": "paraphrase-multilingual-mpnet-base-v2", "model_size": 768, "is_bert": True},
    {"model_path": "intfloat/e5-base-v2", "model_size": 768, "is_bert": True},
    {"model_path": "all-roberta-large-v1", "model_size": 1024, "is_bert": True},
    {"model_path": "dangvantuan/sentence-camembert-base", "model_size": 768, "is_bert": True},
    {"model_path": "OrdalieTech/Solon-embeddings-large-0.1", "model_size": 1024, "is_bert": True},
    {"model_path": "FacebookAI/xlm-roberta-large", "model_size": 1024, "is_bert": True},
    {"model_path": "distilbert/distilbert-base-uncased", "model_size": 768, "is_bert": True},
    {"model_path": "sentence-transformers/all-MiniLM-L12-v2", "model_size": 384, "is_bert": True},
    {"model_path": "intfloat/multilingual-e5-large", "model_size": 1024, "is_bert": True},
    {"model_path": "models/text-embedding-004", "model_size": 768, "embedding_func": get_text_embedding_genai}, 
#    {"model_path": "voyage-2", "model_size": 1024, "embedding_func": get_text_embedding_voyageai}, 
]

# To stay under the rate limits (for instance 22 secondes for voyage, 2s for the other models, but depends on the subscriptions plans) 
time_sleep = 2
# time_sleep = 22
##









# save and load embeddings using pickle
def save_embeddings_with_pickle(embeddings, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    # Construct the file path
    file_path = os.path.join(save_dir, f"{safe_model_name}_embeddings.pkl")    
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {file_path}")

def save_file_names_with_pickle(file_names, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    # Construct the file path
    file_path = os.path.join(save_dir, f"{safe_model_name}_file_names.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(file_names, f)
    print(f"File names saved to {file_path}")

# function to load text files, handle encoding, and compute embeddings
def load_and_compute_embeddings(list_texts, model_size, get_text_embedding_func=None, model_path=None, is_bert=False):
    embeddings = np.zeros([len(list_texts), model_size])

    if is_bert:
        model = SentenceTransformer(model_path, trust_remote_code=True)

    for i, (file_name, dir_key, path) in enumerate(list_texts):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                sentences = [f.read()]
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='latin-1') as f:  # Try an alternative encoding
                    sentences = [f.read()]
            except UnicodeDecodeError:
                with open(path, 'r', encoding='ISO-8859-1') as f:  # Another fallback encoding
                    sentences = [f.read()]

        print(file_name)
        time.sleep(time_sleep)
        
        if is_bert:
            embeddings[i, :] = model.encode(sentences)
        else:
            embeddings[i, :] = get_text_embedding_func(sentences[0], model_path)
    
    return embeddings



# workflow for each class
for class_name, path in all_paths.items():
    # glob gets the list of .txt files and sort them alphabetically
    list_texts = [(os.path.basename(file), class_name, file) for file in sorted(glob.glob(os.path.join(path, '*.txt')))]
    file_names = [file for file, _, _ in list_texts]  # List of file names
    
    for config in model_configs:
        print(config)
        model_path = config['model_path']
        model_size = config['model_size']
        is_bert = config.get('is_bert', False)
        embedding_func = config.get('embedding_func', None)
        
        embeddings = load_and_compute_embeddings(list_texts, model_size, get_text_embedding_func=embedding_func, model_path=model_path, is_bert=is_bert)
        
        # Save embeddings with pickle in a directory specific to the class
        # save_embeddings_with_pickle(embeddings, model_path, save_dir=f"embeddings_{class_name}")
        print(f"embeddings_{class_name}")
        
        # Save file names corresponding to the embeddings
        # save_file_names_with_pickle(file_names, model_path, save_dir=f"embeddings_{class_name}")