# Exploring Writing Style in Embedding Space: Human Literary Texts versus Language Model Generations

## Repository for the data and experiments accompanying our LREC 2026 submission

This repository contains all the code, data, and results associated with the paper: Exploring Writing Style in Embedding Space: Human Literary Texts versus Language Model Generations. This repository's purpose is to generate embeddings, reproduce our analyses, and explore the results.

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

Some models require API keys to use. In you want to use one of these, you should add your API key in the config.yaml file.

All experiments using the embeddings provided can be done without API keys, and we provided several embeddings models which don't require them either.

## Dataset

In the Texts subfolder: STYLE_GEN (subset of StyloGen, not fully available) comprises 864 French texts created by prompting three large language models (GPT-4o, mistral-large-2411, and gemini-1.5-flash) to rewrite the 96 fixed-topic passages from Stéphane Tufféry's Le style mode d'emploi (2000) in the style of three canonical French authors: Marcel Proust's Du côté de chez Swann (1913), Louis-Ferdinand Céline's Voyage au bout de la nuit (1932), and Marguerite Yourcenar's Mémoires d'Hadrien (1951). All generated texts preserve the same narrative content, namely the Paris bus-journey episode adapted from Raymond Queneau's Exercices de style (1947), while varying only in stylistic rendition according to the target author and model. These constitute transformative stylistic imitations consistent with fair use principles. This controlled design enables systematic analysis of how embeddings represent and differentiate authorial style independently of topical content.

In the Embeddings subfolder: The StyloGen set consists of 16,224 vector embeddings derived from 1,248 French texts in the StyloGen corpus, each represented across 13 state-of-the-art embedding models. The underlying texts include 384 human-authored literary passages (from Stéphane Tufféry's Le style mode d'emploi (2000), Marcel Proust's Du côté de chez Swann (1913), Louis-Ferdinand Céline's Voyage au bout de la nuit (1932), and Marguerite Yourcenar's Mémoires d'Hadrien (1951)) and 864 machine-generated rewritings created by GPT-4o, mistral-large-2411, and gemini-1.5-flash in the styles of these authors. Each text was embedded using thirteen multilingual or French-compatible models selected for diversity and high performance on the Massive Text Embedding Benchmark (MTEB), including xlm-roberta-large, mistral-embed, text-embedding-3-small, text-embedding-004, multilingual-e5-large, and others.

The resulting collection contains 16,224 non-reversible embedding vectors (1,248 texts × 13 models), released for research use under a non-commercial, research-only license. These embeddings allow quantitative investigation of how different vectorization models encode and differentiate writing style, topic, and authorship signals in both human and LLM-generated texts.


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

We provided the texts in .txt format for the STYLE_GEN class, but not for the TUFFERY_REF (Stéphane Tufféry. 2000. Le style mode d’emploi. Paris: Cylibris) or STYLE_REF (Marcel Proust. 1913. Du côté de chez Swann, volume 1, chapter 1. Gallimard; Louis-Ferdinand Céline. 1932. Voyage au bout de la nuit. Denoël et Steele; Marguerite Yourcenar. 1951. Mémoires d’Hadrien. Plon) classes.


### Exemple use: 
From style-embedding-sensitivity:
```
 python Embeddings/3_embeddings_creation.py
```