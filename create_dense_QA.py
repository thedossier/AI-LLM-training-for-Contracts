import subprocess
import numpy as np

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def run_create_QA_script(times, chunks_to_process, api_choice, context, train_output_filename, test_output_filename):
    temperatures = np.linspace(0.01, 1.2, times)
    for temperature in temperatures:
        print(f"Running script with temperature: {temperature}, process option: {chunks_to_process}, API choice: {api_choice} and context: {context}")
        subprocess.run(["python3", "create_QA.py", "--api", api_choice, "--temperature", str(temperature), "--chunks_to_process", chunks_to_process, "--context", str(context), "--train_output_filename", str(train_output_filename),  "--test_output_filename", str(test_output_filename)])
        print(f"Completed run with temperature: {temperature}\n")

if __name__ == "__main__":
    # Function to create or empty a file
    train_output_filename="train.txt"
    test_output_filename="test.txt"
    def reset_file(file_path):
        with open(f"./data/{file_path}", "w", encoding='utf-8') as file:
            pass

    reset_file(train_output_filename)
    reset_file(test_output_filename)

    context = input("Enter one sentence to provide the context of the dataset you are training on (e.g. International Touch Rugby Rules): ")
    times = int(input("Enter the number of times to iterate over QA generation (Recommend 1 for test, 9 for production): ").strip())
    chunks_to_process = input("Do you want to process one chunk (for testing) or all chunks? Enter 'one' or 'all': ").strip().lower()
    api_choice = input("Choose the API to use (openai/runpod): ").strip().lower()
    run_create_QA_script(times, chunks_to_process, api_choice, context, train_output_filename, test_output_filename)
