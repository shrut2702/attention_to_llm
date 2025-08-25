import json
import urllib.request
import psutil
from tqdm import tqdm

def load_json_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

def format_entry(entry):
    instruction_text = (
        f"Below is an instruction that describes a task."
        f" Write a response that address the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )    
    input_text = f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    return instruction_text + input_text

def query_model(prompt, model="llama3", url="http://localhost:11435/api/chat"):
    data = {
        "model":model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode('utf-8')
    request = urllib.request.Request(url,data=payload, method='POST')
    request.add_header('Content-Type', 'application/json')

    response_data =""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode('utf-8')
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json['message']['content']

    return response_data

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(['name']):
        if process_name in proc.info['name']:
            running = True
            break
    return running


if __name__ == "__main__":
    json_file = 'instruction-data-with-response-lora-rank16.json'
    ollama_running = check_if_running("ollama")
    if not ollama_running:
        raise RuntimeError("Ollama server is not running. Please start it before running this script.")
    
    response_entries = load_json_responses(f"../Data/{json_file}")

    scores = []

    for entry in tqdm(response_entries, desc="Processing entries", unit="entry"):
        prompt = (
            f"Given the following the task: {format_entry(entry)} "
            f"and the correct response: {entry['output']}, "
            f"the model's response is {entry['model_response']}. "
            f"Please evaluate the model's response and provide a score between 0 and 100, 100 being the best score. "
            f"Return only integer score without any additional text."
        )

        try:
            scores.append(int(query_model(prompt)))
        except ValueError:
            print(f"Error processing entry: {entry['instruction']}. Invalid score returned.")
            continue

    average_score = sum(scores)/len(scores) if scores else 0
    print(f"Number of entries scored: {len(scores)}")
    print(f"Average score for the model's responses: {average_score:.2f}")

    