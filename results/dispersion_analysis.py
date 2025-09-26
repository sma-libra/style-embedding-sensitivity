# pip install numpy pandas scipy umap-learn pickle5

import numpy as np
import pandas as pd
import os
import pickle
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from scipy.stats import chi2
import umap


# Base paths for the datasets in French and English
# your_path = "./Mistral/tuffery_yourcenar/embeddings/"

base_paths = {
    "FR": f"./french/embeddings/",
    "EN": f"./english/embeddings/"
}


# Prompt user to select language
language = input("Choose language (FR/EN): ").upper()
if language not in base_paths:
    raise ValueError("Invalid language choice. Please select 'FR' or 'EN'.")

# author = input("Choose author (proust/celine/yourcenar): ").lower()
# if author not in ['proust', 'celine', 'yourcenar']:
#     raise ValueError("Invalid author choice.")


# model_gen = input("Choose Gen model (deepseek/gemini/gpt/mistral): ").lower()
# if model_gen not in ['deepseek', 'gemini', 'gpt', 'mistral']:
#     raise ValueError("Invalid model gen choice.")

# Output paths for results and plots in French and English
results_dirs = {
    # "FR": f'./french/results/dispersion/{author}_{model_gen}/',
    "FR": f'./french/results/dispersion/B1_A2/',
    "EN": f'./english/results/dispersion/B1_A2/',
}

# Set base path and output directories based on the selected language
base_path = base_paths[language]
results_dir = results_dirs[language]
# output_file = os.path.join(results_dir, f"clustering_evaluation_results_{language}_{author}_{model_gen}.xlsx")
plots_dir = os.path.join(results_dir, "plots")

# Ensure results and plots directories exist
os.makedirs(results_dir, exist_ok=True)
# os.makedirs(plots_dir, exist_ok=True)

# Datasets paths
if language == "FR":
    dataset_folders = {
        # "B1_renard" : "B1_renard", 
        # "A1" : "A1_fr", 
        # "A2" : f"A2_fr/A2_{model_gen}_{author}_fr/", 
        # "B" : f"B_fr/B_{author}_fr"
        # "Tuffery":"A1_fr", 
        "Proust": "B_fr/B_proust_fr",
        "Celine": "B_fr/B_celine_fr", 
        "Yourcenar":"B_fr/B_yourcenar_fr",
        "Proust_mistral": "A2_fr/A2_mistral_proust_fr",
        "Celine_mistral": "A2_fr/A2_mistral_celine_fr", 
        "Yourcenar_mistral":"A2_fr/A2_mistral_yourcenar_fr",
        "Proust_gemini": "A2_fr/A2_gemini_proust_fr",
        "Celine_gemini": "A2_fr/A2_gemini_celine_fr", 
        "Yourcenar_gemini":"A2_fr/A2_gemini_yourcenar_fr",
        "Proust_gpt": "A2_fr/A2_gpt_proust_fr",
        "Celine_gpt": "A2_fr/A2_gpt_celine_fr", 
        "Yourcenar_gpt":"A2_fr/A2_gpt_yourcenar_fr",
    }
elif language == "EN":
    dataset_folders = {
        # "B1_renard" : "B1_renard", 
        # "A1" : "A1_en", 
        # "A2" : f"A2_en/A2_{model_gen}_{author}_en/", 
        # "B" : f"B_en/B_{author}_en"
        # "Tuffery":"A1_en", 
        "Proust": "B_en/B_proust_en",
        "Celine": "B_en/B_celine_en", 
        "Yourcenar":"B_en/B_yourcenar_en",
        "Proust_mistral": "A2_en/A2_mistral_proust_en",
        "Celine_mistral": "A2_en/A2_mistral_celine_en", 
        "Yourcenar_mistral":"A2_en/A2_mistral_yourcenar_en",
        "Proust_gemini": "A2_en/A2_gemini_proust_en",
        "Celine_gemini": "A2_en/A2_gemini_celine_en", 
        "Yourcenar_gemini":"A2_en/A2_gemini_yourcenar_en",
        "Proust_gpt": "A2_en/A2_gpt_proust_en",
        "Celine_gpt": "A2_en/A2_gpt_celine_en", 
        "Yourcenar_gpt":"A2_en/A2_gpt_yourcenar_en",
    }


# Predefined UMAP dimensions and seeds
# umap_dimensions = [2, 3, 5, 10]
umap_dimensions = [2]
predefined_seeds = [42, 7, 19, 23, 1, 100, 56, 77, 89, 33, 8, 
                    13, 5, 21, 34, 99, 67, 18, 50, 81, 45, 22, 74, 37, 58, 
                    90, 16, 11, 29, 85]
# predefined_seeds = [2]

# Model configurations
model_configs = [
    {"model_name": "mistral-embed"},
    {"model_name": "text-embedding-3-small"},
    {"model_name": "voyage-2"},
    {"model_name": "paraphrase-multilingual-mpnet-base-v2"},
    {"model_name": "intfloat/e5-base-v2"},
    {"model_name": "all-roberta-large-v1"},
    {"model_name": "dangvantuan/sentence-camembert-base"},
    {"model_name": "OrdalieTech/Solon-embeddings-large-0.1"},
    {"model_name": "FacebookAI/xlm-roberta-large"},
    {"model_name": "distilbert/distilbert-base-uncased"},
    {"model_name": "sentence-transformers/all-MiniLM-L12-v2"},
    {"model_name": "intfloat/multilingual-e5-large"},
    {"model_name": "models/text-embedding-004"}, 
]

# Function to load embeddings from pickle files
def load_embeddings_from_pickle(embeddings_dir):
    try:
        with open(embeddings_dir, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except FileNotFoundError as e:
        print(f"File not found: {embeddings_dir}")
        raise e

# Function to calculate centroid and distances
def calculate_centroid_and_distances(embeddings):
    centroid = np.mean(embeddings, axis=0)
    distances = cdist(embeddings, [centroid]).flatten()
    return centroid, distances

# Function to calculate mean distances for all predefined seeds and UMAP dimensions
def calculate_mean_distances(embeddings_proust, embeddings_celine, embeddings_yourcenar, 
                             embeddings_proust_gpt, embeddings_celine_gpt, embeddings_yourcenar_gpt, 
                             embeddings_proust_mistral, embeddings_celine_mistral, embeddings_yourcenar_mistral, 
                             embeddings_proust_gemini, embeddings_celine_gemini, embeddings_yourcenar_gemini, n_components):
    all_distances_proust, all_distances_celine, all_distances_yourcenar, all_distances_proust_gpt, all_distances_celine_gpt, all_distances_yourcenar_gpt, all_distances_proust_mistral, all_distances_celine_mistral, all_distances_yourcenar_mistral, all_distances_proust_gemini, all_distances_celine_gemini, all_distances_yourcenar_gemini = [], [], [], [], [], [], [], [], [], [], [], []
    
    for seed in predefined_seeds:
        reducer = umap.UMAP(n_components=n_components, random_state=seed, n_jobs=1)
        all_embeddings = np.concatenate((embeddings_proust, embeddings_celine, embeddings_yourcenar, 
                                         embeddings_proust_gpt, embeddings_celine_gpt, embeddings_yourcenar_gpt, 
                             embeddings_proust_mistral, embeddings_celine_mistral, embeddings_yourcenar_mistral, 
                             embeddings_proust_gemini, embeddings_celine_gemini, embeddings_yourcenar_gemini), axis=0)
        reducer.fit(all_embeddings)
        
        transformed_proust = reducer.transform(embeddings_proust)
        transformed_celine = reducer.transform(embeddings_celine)
        transformed_yourcenar = reducer.transform(embeddings_yourcenar)
        transformed_proust_gpt = reducer.transform(embeddings_proust_gpt)
        transformed_celine_gpt = reducer.transform(embeddings_celine_gpt)
        transformed_yourcenar_gpt = reducer.transform(embeddings_yourcenar_gpt)
        transformed_proust_mistral = reducer.transform(embeddings_proust_mistral)
        transformed_celine_mistral = reducer.transform(embeddings_celine_mistral)
        transformed_yourcenar_mistral = reducer.transform(embeddings_yourcenar_mistral)
        transformed_proust_gemini = reducer.transform(embeddings_proust_gemini)
        transformed_celine_gemini = reducer.transform(embeddings_celine_gemini)
        transformed_yourcenar_gemini = reducer.transform(embeddings_yourcenar_gemini)
        
        _, distances_proust = calculate_centroid_and_distances(transformed_proust)
        _, distances_celine = calculate_centroid_and_distances(transformed_celine)
        _, distances_yourcenar = calculate_centroid_and_distances(transformed_yourcenar)
        _, distances_proust_gpt = calculate_centroid_and_distances(transformed_proust_gpt)
        _, distances_celine_gpt = calculate_centroid_and_distances(transformed_celine_gpt)
        _, distances_yourcenar_gpt = calculate_centroid_and_distances(transformed_yourcenar_gpt)
        _, distances_proust_mistral = calculate_centroid_and_distances(transformed_proust_mistral)
        _, distances_celine_mistral = calculate_centroid_and_distances(transformed_celine_mistral)
        _, distances_yourcenar_mistral = calculate_centroid_and_distances(transformed_yourcenar_mistral)
        _, distances_proust_gemini = calculate_centroid_and_distances(transformed_proust_gemini)
        _, distances_celine_gemini = calculate_centroid_and_distances(transformed_celine_gemini)
        _, distances_yourcenar_gemini = calculate_centroid_and_distances(transformed_yourcenar_gemini)
        
        all_distances_proust.append(distances_proust)
        all_distances_celine.append(distances_celine)
        all_distances_yourcenar.append(distances_yourcenar)
        all_distances_proust_gpt.append(distances_proust_gpt)
        all_distances_celine_gpt.append(distances_celine_gpt)
        all_distances_yourcenar_gpt.append(distances_yourcenar_gpt)
        all_distances_proust_mistral.append(distances_proust_mistral)
        all_distances_celine_mistral.append(distances_celine_mistral)
        all_distances_yourcenar_mistral.append(distances_yourcenar_mistral)
        all_distances_proust_gemini.append(distances_proust_gemini)
        all_distances_celine_gemini.append(distances_celine_gemini)
        all_distances_yourcenar_gemini.append(distances_yourcenar_gemini)
    
    mean_distances_proust = np.mean([np.mean(d) for d in all_distances_proust])
    mean_distances_celine = np.mean([np.mean(d) for d in all_distances_celine])
    mean_distances_yourcenar = np.mean([np.mean(d) for d in all_distances_yourcenar])
    mean_distances_proust_gpt = np.mean([np.mean(d) for d in all_distances_proust_gpt])
    mean_distances_celine_gpt = np.mean([np.mean(d) for d in all_distances_celine_gpt])
    mean_distances_yourcenar_gpt = np.mean([np.mean(d) for d in all_distances_yourcenar_gpt])
    mean_distances_proust_mistral = np.mean([np.mean(d) for d in all_distances_proust_mistral])
    mean_distances_celine_mistral = np.mean([np.mean(d) for d in all_distances_celine_mistral])
    mean_distances_yourcenar_mistral = np.mean([np.mean(d) for d in all_distances_yourcenar_mistral])
    mean_distances_proust_gemini = np.mean([np.mean(d) for d in all_distances_proust_gemini])
    mean_distances_celine_gemini = np.mean([np.mean(d) for d in all_distances_celine_gemini])
    mean_distances_yourcenar_gemini = np.mean([np.mean(d) for d in all_distances_yourcenar_gemini])

    return (mean_distances_proust, mean_distances_celine, mean_distances_yourcenar, 
            mean_distances_proust_gpt, mean_distances_celine_gpt, mean_distances_yourcenar_gpt, 
            mean_distances_proust_mistral, mean_distances_celine_mistral, mean_distances_yourcenar_mistral,  
            mean_distances_proust_gemini, mean_distances_celine_gemini, mean_distances_yourcenar_gemini, 
            np.array(all_distances_proust), np.array(all_distances_celine), np.array(all_distances_yourcenar), 
            np.array(all_distances_proust_gpt), np.array(all_distances_celine_gpt), np.array(all_distances_yourcenar_gpt), 
            np.array(all_distances_proust_mistral), np.array(all_distances_celine_mistral), np.array(all_distances_yourcenar_mistral), 
            np.array(all_distances_proust_gemini), np.array(all_distances_celine_gemini), np.array(all_distances_yourcenar_gemini)
    )
# Main execution
all_model_distances = {class_name: [] for class_name in dataset_folders.keys()}

for config in model_configs:
    model_name = config['model_name']
    print(model_name)
    
    # Load the embeddings for each class
    embeddings_dict, distances_dict = {}, {}
    for class_name, folder in dataset_folders.items():
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        embeddings_dir = os.path.join(base_path, folder, f"{safe_model_name}_embeddings.pkl")
        embeddings_dict[class_name] = load_embeddings_from_pickle(embeddings_dir)
        
        # Calculate distances from the centroid
        _, distances = calculate_centroid_and_distances(embeddings_dict[class_name])
        distances_dict[class_name] = distances
    
    # embeddings_A1, embeddings_B1 = embeddings_dict["A1_tuffery"], embeddings_dict["B1_renard"]
    # embeddings_A2, embeddings_B2 = embeddings_dict["A2"], embeddings_dict["B2"]

    embeddings_proust = embeddings_dict["Proust"]
    embeddings_celine = embeddings_dict["Celine"]
    embeddings_yourcenar = embeddings_dict["Yourcenar"]
    embeddings_proust_gpt = embeddings_dict["Proust_gpt"]
    embeddings_celine_gpt = embeddings_dict["Celine_gpt"]
    embeddings_yourcenar_gpt = embeddings_dict["Yourcenar_gpt"]
    embeddings_proust_mistral = embeddings_dict["Proust_mistral"]
    embeddings_celine_mistral = embeddings_dict["Celine_mistral"]
    embeddings_yourcenar_mistral = embeddings_dict["Yourcenar_mistral"]
    embeddings_proust_gemini = embeddings_dict["Proust_gemini"]
    embeddings_celine_gemini = embeddings_dict["Celine_gemini"]
    embeddings_yourcenar_gemini = embeddings_dict["Yourcenar_gemini"]
    
    results = []
    for n_components in umap_dimensions:
        mean_dist_proust, mean_dist_celine, mean_dist_yourcenar, mean_dist_proust_gpt, mean_dist_celine_gpt, mean_dist_yourcenar_gpt, mean_dist_proust_mistral, mean_dist_celine_mistral, mean_dist_yourcenar_mistral, mean_dist_proust_gemini, mean_dist_celine_gemini, mean_dist_yourcenar_gemini, all_distances_proust, all_distances_celine, all_distances_yourcenar, all_distances_proust_gpt, all_distances_celine_gpt, all_distances_yourcenar_gpt, all_distances_proust_mistral, all_distances_celine_mistral, all_distances_yourcenar_mistral, all_distances_proust_gemini, all_distances_celine_gemini, all_distances_yourcenar_gemini = calculate_mean_distances(
             embeddings_proust, embeddings_celine, embeddings_yourcenar, 
             embeddings_proust_gpt, embeddings_celine_gpt, embeddings_yourcenar_gpt, 
            embeddings_proust_mistral, embeddings_celine_mistral, embeddings_yourcenar_mistral,
             embeddings_proust_gemini, embeddings_celine_gemini, embeddings_yourcenar_gemini, n_components
        )

        if n_components == 2:
            for class_name, distances in zip(dataset_folders.keys(), [all_distances_proust, all_distances_celine, all_distances_yourcenar, 
                                                                      all_distances_proust_gpt, all_distances_celine_gpt, all_distances_yourcenar_gpt, 
                                                                      all_distances_proust_mistral, all_distances_celine_mistral, all_distances_yourcenar_mistral, 
                                                                      all_distances_proust_gemini, all_distances_celine_gemini, all_distances_yourcenar_gemini]):
                all_model_distances[class_name].append(np.mean(distances, axis=0))
        
        conditions = [
            # ('B2', 'A1_tuffery', all_distances_B2, all_distances_A1, 'TOPIC'),
            # ('B', 'A2', all_distances_B, all_distances_A2, 'TOPIC'),
            # ('A1', 'A2', all_distances_A1, all_distances_A2, 'STYLE'),
            # # ('B2', 'B1_renard', all_distances_B2, all_distances_B1, 'STYLE'),
            # ('B', 'A1', all_distances_B, all_distances_A1, 'GLOBAL')
        ]
        
    #     for group1_name, group2_name, distances1, distances2, cond_type in conditions:
    #         t_stat, p_value = ttest_ind(np.hstack(distances1), np.hstack(distances2), equal_var=False)
    #         mean1 = np.mean(np.hstack(distances1))
    #         mean2 = np.mean(np.hstack(distances2))
    #         condition_satisfied = mean1 > mean2
    #         results.append({
    #             'Model': model_name,
    #             'UMAP_Dimension': f'UMAP_{n_components}D',
    #             'Condition': f'{group1_name} > {group2_name}',
    #             'Condition_Type': cond_type,
    #             'Group1': group1_name,
    #             'Group2': group2_name,
    #             'Mean1': mean1,
    #             'Mean2': mean2,
    #             'Condition_Satisfied': condition_satisfied,
    #             'T_Statistic': t_stat,
    #             'P_Value_T': p_value,
    #             'Condition_Significant': p_value < 0.05 if not np.isnan(p_value) else False
    #         })

    # results_df = pd.DataFrame(results)
    # # Save results per model
    # safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    # results_file = os.path.join(results_dir, f'results_{safe_model_name}_global_{language}.xlsx')
    # results_df.to_excel(results_file, index=False)
    # print(f"Results for model {model_name} saved to {results_file}")

# Calculate and save the mean distances across all models for UMAP 2D only
mean_distances_data = []
for class_name, distances_list in all_model_distances.items():
    distances_array = np.array(distances_list)
    mean_distances = np.mean(distances_array, axis=0)
    for idx, mean_distance in enumerate(mean_distances):
        mean_distances_data.append({
            'Class': class_name,
            'Text_Index': idx,
            'Mean_Distance_From_Centroid': mean_distance
        })

mean_distances_df = pd.DataFrame(mean_distances_data)
mean_distances_file = os.path.join(results_dir, f'distance_pertext_umap_2d_{language}.xlsx')
mean_distances_df.to_excel(mean_distances_file, index=False)

# Combine all results into one final file
# all_results = []
# for file in os.listdir(results_dir):
#     if file.endswith(f'global_{language}.xlsx') and 'results' in file:
#         file_path = os.path.join(results_dir, file)
#         df = pd.read_excel(file_path)
#         all_results.append(df)

# final_results_df = pd.concat(all_results, ignore_index=True)
# final_results_file = os.path.join(results_dir, f'dispersion_results_{language}.xlsx')
# final_results_df.to_excel(final_results_file, index=False)

# ## Mean & Median Model evaluation 
# final_results_file = os.path.join(results_dir, f'dispersion_results_{language}.xlsx')
# mean_distances_file = os.path.join(results_dir, f'distance_pertext_umap_2d_{language}.xlsx')

# # Load the combined results and mean distances
# final_results_df = pd.read_excel(final_results_file)
# mean_distances_df = pd.read_excel(mean_distances_file)

# # Calculate mean of mean distances for each condition/dimension
# mean_distances_summary = final_results_df.groupby(['UMAP_Dimension', 'Condition']) \
#     .agg({
#         'Mean1': 'mean',
#         'Mean2': 'mean',
#         'Condition_Satisfied': 'mean',
#         'Condition_Significant': 'mean',
#         'P_Value_T': lambda x: chi2.sf(-2 * np.sum(np.log(x)), 2 * len(x))
#     }).reset_index()

# # Calculate median of mean distances for each condition/dimension
# median_mean_distances = final_results_df.groupby(['UMAP_Dimension', 'Condition']) \
#     .agg({
#         'Mean1': 'median',
#         'Mean2': 'median',
#         'Condition_Satisfied': 'median',
#         'Condition_Significant': 'median',
#         'P_Value_T': lambda x: chi2.sf(-2 * np.sum(np.log(x)), 2 * len(x))
#     }).reset_index()

# # Determine if medians and averages are satisfied and significant
# def determine_satisfaction_and_significance(df):
#     df['Condition_Satisfied'] = df['Mean1'] > df['Mean2']
#     df['Condition_Significant'] = df['P_Value_T'] < 0.05
    
#     return df


# mean_distances_summary = determine_satisfaction_and_significance(mean_distances_summary)
# median_mean_distances = determine_satisfaction_and_significance(median_mean_distances)

# # Save the results to new files
# mean_distances_summary_file = os.path.join(results_dir, f'mean_distances_summary_{language}.xlsx')
# median_mean_distances_file = os.path.join(results_dir, f'median_mean_distances_{language}.xlsx')

# mean_distances_summary.to_excel(mean_distances_summary_file, index=False)
# median_mean_distances.to_excel(median_mean_distances_file, index=False)

# print(f"Mean distances summary saved to {mean_distances_summary_file}")
# print(f"Median mean distances saved to {median_mean_distances_file}")

# #### Optimal Dimensionality 

# # Group the final results by UMAP Dimension and calculate counts
# grouped_df = final_results_df.groupby('UMAP_Dimension').agg(
#     Conditions_Satisfied=('Condition_Satisfied', 'sum'),
#     Significant_Conditions=('Condition_Significant', 'sum'),
#     Total_Lines=('Condition_Satisfied', 'size'),
#     Not_Satisfied_And_Significant=('Condition_Significant', lambda x: ((final_results_df['Condition_Satisfied'] == False) & x).sum())
# ).reset_index()

# # Calculate the ratios for each dimensionality
# grouped_df['Ratio_Conditions_Satisfied'] = grouped_df['Conditions_Satisfied'] / grouped_df['Total_Lines']

# # Calculate the ratio of significant conditions to satisfied conditions
# grouped_df['Ratio_Significant_to_Satisfied'] = grouped_df.apply(
#     lambda row: row['Significant_Conditions'] / row['Conditions_Satisfied'] 
#     if row['Conditions_Satisfied'] > 0 else 0, axis=1
# )

# # Calculate the number of not satisfied conditions
# grouped_df['Not_Satisfied_Conditions'] = grouped_df['Total_Lines'] - grouped_df['Conditions_Satisfied']

# # Calculate the ratio of significant conditions that are not satisfied
# grouped_df['Ratio_Significant_to_Not_Satisfied'] = grouped_df.apply(
#     lambda row: row['Not_Satisfied_And_Significant'] / row['Not_Satisfied_Conditions'] 
#     if row['Not_Satisfied_Conditions'] > 0 else 0, axis=1
# )

# # Save the results to an Excel file
# ratios_file = os.path.join(results_dir, f'ratios_per_dimensionality_{language}.xlsx')
# grouped_df.to_excel(ratios_file, index=False)

