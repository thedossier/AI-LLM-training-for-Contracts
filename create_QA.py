import argparse
import os
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import requests
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Setup argument parser
parser = argparse.ArgumentParser(description='Generate QA pairs using specified API, temperature, and process option.')
parser.add_argument('--context', type=str, required=True, help='One sentence describing the context for the dataset.')
parser.add_argument('--train_output_filename', type=str, required=True)
parser.add_argument('--test_output_filename', type=str, required=True)
parser.add_argument('--api', type=str, choices=['openai', 'runpod'], required=True, help='API to use (openai or runpod)')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation (between 0 and 1.2)')
parser.add_argument('--chunks_to_process', type=str, choices=['one', 'all'], required=True, help='Process one chunk or all chunks')
args = parser.parse_args()

# Use arguments
train_output_filename = args.train_output_filename
test_output_filename = args.test_output_filename
api_choice = args.api
temperature = args.temperature
chunks_to_process = args.chunks_to_process
context = args.context

# context
tokens_per_question = 60 # reduce this parameter to increase the granularity of questions. If you reduce this too much the language model may hallucinate content.
# chunk_size = min(model_context_length / (1 + 60/tokens_per_question),25*tokens_per_question)/2 # there are empirically about 60 tokens per QA pair. Also, GPT gets confused making too many questions up.
chunk_size = 2000 # for testing, set a smaller chunk size.
model_context_length = 4000 #This needs to be at least twice the chunk size, otherwise responses will be truncated.

train_sample = "What is the purpose of Section 25.D in the Spirit CBA?\nSection 25.D in the Spirit CBA outlines the Commuter Late Check-In Procedure, which allows Flight Attendants who are unable to report by their designated check-in time but arrive before departure to take their original trip as long as it is not anticipated to result in a flight delay."
test_sample = "What is the purpose of Section 25.D in the Spirit CBA?\nSection 25.D in the Spirit CBA outlines the Commuter Late Check-In Procedure, which allows Flight Attendants who are unable to report by their designated check-in time but arrive before departure to take their original trip as long as it is not anticipated to result in a flight delay."

questions_per_chunk_train = int(chunk_size / tokens_per_question)
questions_per_chunk_test = max(int(questions_per_chunk_train / 10),1)

print(f'Setting {questions_per_chunk_train} questions per {int(chunk_size)}-token chunk for QA train dataset generation.')

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = sum(1 for _ in encoding.encode(text))
    return token_count

def read_and_chunk_txt(file_path):
    chunks = []
    chunk = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if count_tokens(chunk + text) > chunk_size:
                chunks.append(chunk.strip())
                chunk = text
            else:
                chunk += " " + text
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def query_runpod(pod_id, prompt, max_tokens, temperature):
    url = f"https://{pod_id}-8080.proxy.runpod.net/generate"
    prompt = f'[INST] {prompt} [/INST]\n\n'
    # print(f"Prompt in query runpod: {prompt}")
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop": [
                "</s>",
                "[INST]"
            ]
        }
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    # print(f"Status Code: {response.status_code}")
    # print(f"Raw runpod response: {response.text}")
    if response.status_code == 200:
        return json.loads(response.text)["generated_text"]
    else:
        return None

load_dotenv()

# Set up API client


# if api_choice == "openai":
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     if not openai.api_key:
#         print("OpenAI API key is missing. Exiting.")
#         exit(1)
# elif api_choice == "runpod":
#     pod_id = os.getenv("RUNPOD_POD_ID")
#     print(f"Pod id is : {pod_id}")
#     if not pod_id:
#         print("RunPod Pod ID is missing. Exiting.")
#         exit(1)
# else:
#     print("Invalid API choice. Exiting.")
#     exit(1)

chunks = read_and_chunk_txt("data/raw_train.txt")

total_tokens = sum(count_tokens(chunk) for chunk in chunks)
print(f"Total tokens in all chunks: {total_tokens}")

estimated_input_tokens = total_tokens * 1.1
estimated_output_tokens = total_tokens * 50/tokens_per_question
total_estimated_tokens = estimated_input_tokens + estimated_output_tokens

estimated_cost_gpt4 = (estimated_input_tokens / 1000 * 0.03) + (estimated_output_tokens  / 1000 * 0.06)
estimated_cost_gpt35turbo = (estimated_input_tokens / 1000 * 0.0005) + (estimated_output_tokens  / 1000 * 0.0015)
print(f"Estimated cost with gpt-4: ${estimated_cost_gpt4:.2f}")
print(f"Estimated cost with gpt-3.5-turbo-16k: ${estimated_cost_gpt35turbo:.2f}")

while True:
    if chunks_to_process in ['one', 'all']:
        break
    else:
        print("Invalid option. Please enter 'one' or 'all'.")

snippets = [
    f"Provide {questions_per_chunk_train} question and answer pair(s) based on the text above. The question should include sufficient information for the answer, without the user having any further context. The answers need not necessarily borrow verbatim from the input text, but they should maintain the meaning. Vary the style and format of questions. Include some tricky and nuanced questions. In certain answers, reverse the order of words compared to how they appear in the input text. Respond in plain text on a new line for each question and answer. Do not include question numbers. Here is an example of a question answer pair:\n\n<example>\n\n{train_sample}\n\n</example>\n\n",
    f"Provide {questions_per_chunk_test} question and answer pair(s) based on the text above. The question should include sufficient information for the answer, without the user having any further context. The answers should NOT borrow verbatim from the text above, but they should maintain the meaning. Vary the style and format of questions. Respond in plain text on a new line for each question and answer. Do not include question numbers. Here is an example of a question answer pair:\n\n<example>\n\n{test_sample}\n\n</example>\n\n"
]

# Function to validate the response format
def is_valid_qa_format(text):
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Check for at least one question and one answer
    has_question = any(line.endswith('?') for line in lines)
    # has_answer = len(lines) > 1 and any(not line.endswith('?') for line in lines)  # Assuming answer lines don't end with a question mark
    
    # Basic syntax safety check (example: avoid null bytes which can be problematic)
    syntax_safe = all('\0' not in line for line in lines)
    
    return has_question and syntax_safe

def clean_text(text):
    # Remove control characters except newline
    text = re.sub(r'[\x00-\x09\x0b-\x1F\x7F]', '', text)
    
    # Escape potentially dangerous characters or sequences
    # This is an example; adapt based on your context
    text = re.sub(r'[<>{};`]', '', text)
    
    return text

# Function to log errors
def log_error(prompt, response, filename="error_log.txt"):
    with open(filename, "a", encoding='utf-8') as error_file:
        error_file.write(f"Prompt: {prompt}\nResponse: {response}\n\n")

# Update your loop for processing chunks
for idx, snippet in enumerate(snippets):
    # print(snippet)
    output_filename = f"data/{train_output_filename}" if idx == 0 else f"data/{test_output_filename}"
    
    with open(output_filename, "a", encoding='utf-8') as output_file:
        if api_choice == "openai":
            for chunk_idx, chunk in enumerate(chunks):
                prompt = f"<input-text>\n\nContext: {context}\n\nText:\n\n{chunk}\n\n</input-text>\n\n{snippet}"
                if chunks_to_process == 'one' and chunk_idx > 0:
                    break
                if chunks_to_process == 'one':
                    print(f"The prompt is:\n\n{prompt}")

                completion = client.chat.completions.create(
                    model="TheBloke/vicuna-13B-v1.5-16K-GGUF",
                    temperature=temperature,
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                )
                max_tokens=int(model_context_length * 0.9) # Ensure you have defined max_tokens appropriately)

                response = completion.choices[0].message.content
                # print(f"\n\nRaw Response:\n\n{response}")
                
                cleaned_response = clean_text(response)

                # print(f"\n\nCleaned Response:\n\n{cleaned_response}")

                if is_valid_qa_format(cleaned_response):
                    output_file.write(cleaned_response + "\n\n")
                else:
                    log_error(prompt, cleaned_response, filename=f"error_log_{idx}.txt") # Log the error with a unique file per snippet

                output_file.flush()
                time.sleep(0.2)

        elif api_choice == "runpod":
            max_tokens = int(model_context_length * 0.9)
            if chunks_to_process == 'all':
                with ThreadPoolExecutor(max_workers=8) as executor:
                    prompt = f"<input-text>\n\nContext: {context}\n\nText:\n\n{chunk}\n\n</input-text>\n\n{snippet}"
                    future_to_chunk = {executor.submit(query_runpod, pod_id, prompt, max_tokens, temperature): chunk for chunk in chunks}
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        try:
                            response = future.result()
                        except Exception as exc:
                            print(f"Generated an exception: {exc}")
                        else:
                            if is_valid_qa_format(response):
                                output_file.write(response + "\n\n")
                            else:
                                log_error(chunk, response, filename=f"error_log_{idx}.txt") # Log the error with a unique file per snippet

                            output_file.flush()
            else:
                for chunk in chunks:
                    prompt = f"<input-text>\n\nContext: {context}\n\nText:\n\n{chunk}\n\n</input-text>\n\n{snippet}"
                    # print(f"Prompt:\n\n{prompt}")
                    response = query_runpod(pod_id, prompt, max_tokens, temperature)
                    # print(f"\n\nResponse:\n\n{response}")
                    cleaned_response = clean_text(response)
                    # print(f"Cleaned Response:\n\n{cleaned_response}")
                    if is_valid_qa_format(cleaned_response):
                        output_file.write(cleaned_response + "\n\n")
                    else:
                        log_error(chunk, cleaned_response, filename=f"error_log_{idx}.txt")

                    output_file.flush()

                    if chunks_to_process == 'one':
                        break


