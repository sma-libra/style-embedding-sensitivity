# Exploring Writing Style in Embedding Space: Human Literary Texts versus Language Model Generations

## Repository for the experiments and data accompanying our LREC 2026 submission

This repository contains all the code, data, and results associated with the paper: Exploring Writing Style in Embedding Space: Human Literary Texts versus Language Model Generations. This repository's purpose is to reproduce our analyses, generate embeddings, and explore our results.

## Repository structure

```  
├── Embeddings/        
│   └── 3_embeddings_creation.py  
│   └── STYLE_GEN/
│   └── STYLE_REF/
│   └── TUFFERY_REF/
├── Results/                   
│   └── 3.3_clustering-based_validation.ipynb
│   └── 4.1_dispersion.ipynb
│   └── 4.2_vectors_sensitivity_to_human_style.ipynb
│   └── 4.3_stylistic_influence_of_gen_AI.ipynb
│   └── distance_pertext_umap_TS_gen.xlsx
│   └── distance_pertext_umap_TS_ref.xlsx
│   └── stylo_terreau_df.xlsx
├── Texts/     
│   └── STYLE_GEN/   
├── README.md  
├── config.yaml         
```

## Usage

Some models require API keys to use. In you want to use one of these, you should add your API in the config.yaml file.

All experiments using the embeddings provided can be done without API keys, and we provided several embeddings models which don't require them either.

## Results and Notebooks

All analyses corresponding to sections of the article are contained in the Results/ folder as Jupyter notebooks.

Each notebook contains the code to reproduce the results, tables and plots from a specific section of the article. They are named to match the corresponding section of the article.

## Generating Embeddings

We provided the code used to compute the embeddings in 
```
├── Embeddings/                
│   └── 3_embeddings_creation.py  
```
Two variables must be set inside the file to use it : 
- all_paths for the paths of the folder(s) in which the .txt files to be embedded are.
- model_configs for the embedding model(s) to be used. (Note: Some may require an API key)

We provided the texts in .txt format for the STYLE_GEN class, but not for the TUFFERY_REF (Stéphane Tufféry. 2000. Le style mode d’emploi. Paris: Cylibris) ou STYLE_REF (Marcel Proust. 1913. Du côté de chez Swann, volume 1, chapter 1. Gallimard; Louis-Ferdinand Céline. 1932. Voyage au bout de la nuit. Denoël et Steele; Marguerite Yourcenar. 1951. Mémoires d’Hadrien. Plon) classes.


Exemple use: 
```
python Embeddings/3_embeddings_creation.py
```