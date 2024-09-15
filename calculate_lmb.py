import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import os
from scipy import stats
import argparse
import json

def perplexity_score(sentence, model, tokenizer, device):
    """
    Computes the perplexity score of a given sentence using a specified model.
    
    Parameters:
    - sentence: The input sentence for which perplexity is calculated.
    - model: The language model used for calculating perplexity.
    - tokenizer: The tokenizer corresponding to the model.
    - device: The device (CPU/GPU) on which the model will run.
    
    Returns:
    - The perplexity score of the sentence.
    """
    if not isinstance(sentence, str):
        sentence = str(sentence)
    with torch.no_grad():
        model.eval()
        tokenize_input = tokenizer.tokenize(sentence)
        
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
        loss = model(tensor_input, labels=tensor_input)
        return math.exp(loss[0])
def get_perplexity_list(df, m, t, device, demo, task):
    """
    Calculates the perplexity scores for a list of sentences in a DataFrame.
    
    Parameters:
    - df: The DataFrame containing sentences.
    - m: The language model used for calculating perplexity.
    - t: The tokenizer corresponding to the model.
    - device: The device (CPU/GPU) on which the model will run.
    - demo: The demographic group being processed.
    - task: The task type ('intrasentence' or 'intersentence').

    Returns:
    - Two lists containing the perplexity scores for the original and replaced sentences.
    """
    perplexity_list_1 = []
    perplexity_list_2 = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Calculating Perplexity: {demo} ({task})", position=0):
        try:
            if task == 'intrasentence':
                perplexity_1 = perplexity_score(row['sentence'], m, t, device)
                perplexity_2 = perplexity_score(row['replaced_sentence'], m, t, device)
            else:
                perplexity_1 = perplexity_score(f"{row['context']} {row['sentence']}", m, t, device)
                perplexity_2 = perplexity_score(f"{row['replaced_context']} {row['replaced_sentence']}", m, t, device)
        except Exception as ex:
            print(ex.__repr__())
            perplexity_1 = 0
            perplexity_2 = 0
        perplexity_list_1.append(perplexity_1)
        perplexity_list_2.append(perplexity_2)
    return perplexity_list_1, perplexity_list_2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument inputs')

    # Add arguments
    parser.add_argument('--model_id', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--use_cuda', type=str, default='cuda:0', help='Flag to use CUDA if avaliable')
    parser.add_argument('--save_folder', type=str, required=True, help='The folder path where to save the analysis result')

    args = parser.parse_args()

    # Check if the save folder exist, create one if it doesn't.
    if not os.path.exists(args.save_folder):
        print(f"Save folder {args.save_folder} does not exist. Creating it now.")
        os.makedirs(args.save_folder)
        os.makedirs(args.save_folder + '/intersentence', exist_ok=True)
        os.makedirs(args.save_folder + '/intrasentence', exist_ok=True)
        os.makedirs(args.save_folder + '/intersentence/profession', exist_ok=True)
        os.makedirs(args.save_folder + '/intrasentence/profession', exist_ok=True)
        

    # Load the model and tokenizer
    device = args.use_cuda
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)


    demos = ['gender', 'race', 'religion', 'profession']

    # Load the dataset (Make sure new processed dataset is generated)
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'new_data.json')
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for demo in demos:
        if demo == 'profession':
            i = 1
            while i <= 5:
                profession_intersentence_data = data['data']['intersentence']['profession']['profession' + f"{i}"]
                profession_intrasentence_data = data['data']['intrasentence']['profession']['profession' + f"{i}"]

                profession_intersentence_data_df = pd.DataFrame(profession_intersentence_data)
                profession_intrasentence_data_df = pd.DataFrame(profession_intrasentence_data)

                # Calculate perpleixty
                intersentence_perplexity_1, intersentence_perplexity_2 = get_perplexity_list(profession_intersentence_data_df,
                                                                            model, tokenizer, device, 'profession', 'intersentence')
                intrasentence_perplexity_1, intrasentence_perplexity_2 = get_perplexity_list(profession_intrasentence_data_df,
                                                                            model, tokenizer, device, 'profession', 'intrasentence')

                profession_intersentence_data_df['demo1_perplexity'] = intersentence_perplexity_1
                profession_intersentence_data_df['demo2_perplexity'] = intersentence_perplexity_2

                profession_intrasentence_data_df['demo1_perplexity'] = intrasentence_perplexity_1
                profession_intrasentence_data_df['demo2_perplexity'] = intrasentence_perplexity_2

                profession_intrasentence_data_df.to_csv(os.path.join(args.save_folder,
                                                                     f"intrasentence/profession/profession{i}_with_perplexity.csv"),
                                                                     index=False)
                profession_intersentence_data_df.to_csv(os.path.join(args.save_folder, 
                                                                     f"intersentence/profession/profession{i}_with_perplexity.csv"),
                                                                     index=False)

                print(f"Mean and std of perplexities demo1 in profession{i} (Intersentence Task) - Mean {np.mean(profession_intersentence_data_df['demo1_perplexity'])}, Std {np.std(profession_intersentence_data_df['demo1_perplexity'])}")
                print(f"Mean and std of perplexities demo2 in profession{i} (Intrasentence Task) - Mean {np.mean(profession_intrasentence_data_df['demo2_perplexity'])}, Std {np.std(profession_intrasentence_data_df['demo2_perplexity'])}")

                t_value_intersentence, p_value_intersentence = stats.ttest_rel(intersentence_perplexity_1,
                                                  intersentence_perplexity_2)
                t_value_intrasentence, p_value_intrasentence = stats.ttest_rel(intrasentence_perplexity_1,
                                                  intrasentence_perplexity_2)

                print(f'Perplexities of demo1 in profession{i} (Intersentence Task) - T-value {t_value_intersentence} P-valie {p_value_intersentence}')
                print(f'Perplexities of demo1 in profession{i} (Intrasentence Task) - T-value {t_value_intrasentence} P-valie {p_value_intrasentence}')

                result_intersentence = {
                    'Demographic': f"Profession{i}",
                    'Task': 'Intersentence',
                    "T-value": t_value_intersentence,
                    "P-value": p_value_intersentence,
                    "Significance": 'Significant' if p_value_intersentence < 0.05 else 'Not Significant'
                }
                result_intersentence_df = pd.DataFrame([result_intersentence])
                
                result_intrasentence = {
                    'Demographic': f"Profession{i}",
                    'Task': 'Intrasentence',
                    "T-value": t_value_intrasentence,
                    "P-value": p_value_intrasentence,
                    "Significance": 'Significant' if p_value_intrasentence < 0.05 else 'Not Significant'
                }
                result_intrasentence_df = pd.DataFrame([result_intrasentence])
                
                try:
                    result_df = pd.read_csv(os.path.join(args.save_folder, "LMB_result.csv"))
                except FileNotFoundError:
                    result_df = pd.DataFrame(columns=["Demographic", "Task", "T-value", "P-value", "Significance"])

                result_df = pd.concat([result_df, result_intersentence_df], ignore_index=True)
                result_df = pd.concat([result_df, result_intrasentence_df], ignore_index=True)

                result_df.to_csv(os.path.join(args.save_folder, "LMB_result.csv"), index=False)
                print(f"The result of Profession{i} is saved!")
                

                i = i + 1
                
        else:
            intersentence_data = data['data']['intersentence'][demo]
            intrasentence_data = data['data']['intrasentence'][demo]

            intersentence_data_df = pd.DataFrame(intersentence_data)
            intrasentence_data_df = pd.DataFrame(intrasentence_data)

            # Calculate perplexity
            intersentence_perplexity_1, intersentence_perplexity_2 = get_perplexity_list(intersentence_data_df, model, tokenizer,
                                                                                         device, demo, 'intersentence')
            intrasentence_perplexity_1, intrasentence_perplexity_2 = get_perplexity_list(intrasentence_data_df, model, tokenizer,
                                                                                         device, demo, 'intrasentence')
            intersentence_data_df['demo1_perplexity'] = intersentence_perplexity_1
            intersentence_data_df['demo2_perplexity'] = intersentence_perplexity_2

            intrasentence_data_df['demo1_perplexity'] = intrasentence_perplexity_1
            intrasentence_data_df['demo2_perplexity'] = intrasentence_perplexity_2

            intersentence_data_df.to_csv(os.path.join(args.save_folder,
                                                      f"intersentence/{demo}_with_perplexity.csv"), index=False)
            intrasentence_data_df.to_csv(os.path.join(args.save_folder,
                                                      f"intrasentence/{demo}_with_perplexity.csv"), index=False)

            

            print(f"Mean and std of perplexities demo1 in {demo} (Intersentence Task) - Mean {np.mean(intersentence_data_df['demo1_perplexity'])}, Std {np.std(intersentence_data_df['demo1_perplexity'])}\n")
            print(f"Mean and std of perplexities demo2 in {demo} (Intrasentence Task) - Mean {np.mean(intrasentence_data_df['demo2_perplexity'])}, Std {np.std(intrasentence_data_df['demo2_perplexity'])}"),

            t_value_intersentence, p_value_intersentence = stats.ttest_rel(intersentence_perplexity_1,
                                              intersentence_perplexity_2)
            t_value_intrasentence, p_value_intrasentence = stats.ttest_rel(intrasentence_perplexity_1,
                                              intrasentence_perplexity_2)

            print(f'Perplexities of demo1 in {demo} (Intersentence Task) - T-value {t_value_intersentence} P-valie {p_value_intersentence}')
            print(f'Perplexities of demo1 in {demo} (Intrasentence Task) - T-value {t_value_intrasentence} P-valie {p_value_intrasentence}')

            result_intersentence = {
                'Demographic': demo,
                'Task': 'Intersentence',
                "T-value": t_value_intersentence,
                "P-value": p_value_intersentence,
                "Significance": 'Significant' if p_value_intersentence < 0.05 else 'Not Significant'
            }
            result_intersentence_df = pd.DataFrame([result_intersentence])
            
            result_intrasentence = {
                'Demographic': demo,
                'Task': 'Intrasentence',
                "T-value": t_value_intrasentence,
                "P-value": p_value_intrasentence,
                "Significance": 'Significant' if p_value_intrasentence < 0.05 else 'Not Significant'
            }
            result_intrasentence_df = pd.DataFrame([result_intrasentence])
            
            try:
                result_df = pd.read_csv(os.path.join(args.save_folder, "LMB_result.csv"))
            except FileNotFoundError:
                result_df = pd.DataFrame(columns=["Demographic", "Task", "T-value", "P-value", "Significance"])

            result_df = pd.concat([result_df, result_intersentence_df], ignore_index=True)
            result_df = pd.concat([result_df, result_intrasentence_df], ignore_index=True)

            result_df.to_csv(os.path.join(args.save_folder, "LMB_result.csv"), index=False)
            print(f"The result of {demo} is saved!")
                
                
                



