import csv
import json
import os
from transformers import AutoTokenizer

# Function to read questions and answers from a text file and write to a CSV file
def q_and_a_to_csv(input_file_path, output_file_path):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Yarn-Llama-2-7B-128K-GPTQ", use_fast=True)
    
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    messages_list = []  # Initialize an empty list to hold "messages" dictionaries
    
    # Remove empty lines and strip leading/trailing whitespaces
    lines = [line.strip() for line in lines if line.strip()]

    # Process lines to group five interspersed Q&A pairs
    for i in range(0, len(lines), 10):  # Step by 10 to account for 5 Q&A pairs
        messages = {"messages": []}
        for j in range(0, 10, 2):  # Step by 2 within the group of 10 lines
            if i + j + 1 < len(lines):  # Ensure not exceeding list bounds
                user_content = lines[i + j]
                assistant_content = lines[i + j + 1]
                messages["messages"].extend([
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ])
        
        # Convert the dictionary to a JSON string for writing to CSV
        message_json = json.dumps(messages)
        messages_list.append([message_json])

    # Write the list of messages to a CSV file
    with open(output_file_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file, quotechar='"', quoting=csv.QUOTE_ALL)
        csv_writer.writerow(['messages'])  # Write the header row
        csv_writer.writerows(messages_list)

# Specify the paths for the input and output files
train_input_file_path = 'data/train_output.txt'
train_output_file_path = 'data/train_output.csv'

# Run the function to convert train.txt to train.csv
q_and_a_to_csv(train_input_file_path, train_output_file_path)

# Check if test.txt exists, and if so, convert it to text.csv
test_input_file_path = 'data/test_output.txt'
test_output_file_path = 'data/test_output.csv'

if os.path.exists(test_input_file_path):
    q_and_a_to_csv(test_input_file_path, test_output_file_path)
else:
    print("test.txt does not exist, skipping its conversion.")
