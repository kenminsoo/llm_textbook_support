# This is not a script to run. Rather it is a series of bash commands that will get you started.

# To begin, set up a virtual environment, i.e. with conda.

conda create -n llm_forme python=3.9

# Enter the environment
conda activate llm_forme

# Then we need to install required packages
pip install -r requirements.txt

# Now download the model into the models folder
cd models
wget model_link
# model links
# The one used for tests
# https://huggingface.co/TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF/resolve/main/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf

# Then at this point you can go ahead and find a pdf of your textbook
# Then go into one of the scripts (specified in the readme)
# And change parameters at the bottom of the script as needed!