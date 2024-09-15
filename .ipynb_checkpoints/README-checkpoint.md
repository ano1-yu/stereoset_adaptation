# Expanding-Bias-Evaluation-in-LLMs

## Installation
1. Clone the repository: `https://github.com/ano1-yu/stereoset_adaptation.git`
2. Install the requirements: `cd stereoset_adaptation && pip install -r requirements.txt`

## Reproducing Results
To reproduce our results for the bias in each model:

### Languae Modeling Bias
1. Generate adpated dataset based on the original stereoset dataset: `python generate_data.py`
2. Run `calculate_lmb.py` with the following command line:
   ```
   python calculate_lmb.py --model_id ${hugging_face_model_dir} --use_cuda ${using_device} --save_folder ${results_save_address}
   ```

### Toxicity
1. Install Python version of the Google API Client Libraries: `install google-api-python-client` 
2. Run `calculate_toxicity.py` with the following command line:
   ```
   python calculate_toxicity.py --model_id ${hugging_face_model_dir} --use_cuda ${using_device} --save_folder ${results_save_address}
   ```
Notice: Before runing the python file please replace your own perspective api at the line 99 of `calculate_toxicity.py`

### CAT Score
Given that Stereoset was originally released with CAT score we didn't change the official implementation for our evaluation