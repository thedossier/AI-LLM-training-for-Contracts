# Fine-tuning for Content Memorization
Dataset preparation and fine-tuning scripts for memorizing a raw dataset.

> Training for perfect memorization is difficult and requires a strong model and a comprehensive dataset providing knowledge from many different angles.

What can I do?
- Start with a pdf or text
- Convert to a Q&A dataset with a train and test split.
- Run supervised fine-tuning and evaluation using LoRA, QLoRA or full fine-tuning.
- Plot fine-tuning results.
- Push the merged model or adapters to HuggingFace Hub.

Best for:
- Fine-tuning for memorisation.

Training notebook (latest):
- Memorization.ipynb
 
## Getting Started
Activate the python virtual environment on a Mac with:
```
source dataEnv/bin/activate
```
> or 'dataEnv\Scripts\activate' on Windows.

Then install the required packages with:
```
pip install -r requirements.txt
```
Lastly, rename sample.env as .env and put in your OpenAI API key OR runpod endpoint (note that a strong model is required, Mixtral or better).

## Running the Data Preparation Scripts

1. At a minimum, you must have a `train.pdf` (or `raw_train.txt`) file in the `data` folder to get started.
2. Run `python pdf_to_txt.py` to convert PDF files to TXT files (`raw_train.txt` and `raw_test.txt`).
3. To generate a set of questions and answers in `data/train.txt` and `data/test.txt`, use the following command:
   ```bash
   python create_dense_QA.py
   ```
   - `<number_of_iterations>`: The number of times to iterate over QA generation (Recommend 9).
   - `<one|all>`: Choose 'one' to process a single chunk for testing, or 'all' to process all chunks.
   - `<openai|runpod>`: Specify the API to use for generating questions and answers.
4. Run `python QA_to_csv.py` to move `train.txt` and `test.txt` to csv format. The number of tokens per row will be logged to console. Note this for later when you run fine-tuning. The default is for five (5) questions and answers to be combined into the same data row via a conversation.
5. Create a dataset repo on HuggingFace and then run `python push_to_hf.py` to push `train.csv`, `test.csv` and also `data/README.md`, if present.

## Using Llama 2 (Runpod) vs OpenAI
Using OpenAI for prompt generation is not allowed for training competitive language models.

By deploying Llama 2 on a runpod server, you can instead use Llama 2 70B for synthetic data generation. Check out the [one-click-llms repo](github.com/TrelisResearch/one-click-llms) [this video for setup](https://youtu.be/dJ69gY0qRbg).

## Example Details - Touch Rugby Rules

Train.pdf is taken from the [International Touch Website](https://cdn.internationaltouch.org/public/FIT%205th%20Edition%20Rulebook.pdf)

## supervised-fine-tuning-scripts
Typically, you would run these scripts in a separate environment to the data generation.

Most likely this means in Google Colab OR on a service like Runpod or Vast.AI. If you do run on either of Runpod or Vast.AI, make sure to start an instance (with a jupyter notebook) that has Cuda 12.1 or later installed.

For one-click templates and a detailed guide, see the 'main' branch of this repo.

## Using Unsloth
For Llama or Mistral models you can uncomment the cells for installing unsloth and loading the model with unsloth. This gives a 2x training speedup.

## Using quantization
You can use quantization by commenting in the appropriate lines when loading the model. If using quantization (and not unsloth) make sure to uncomment the line that prepares the model for kbit training.

## Changelog

Mar 4 2024:
- Memorization script goes live.