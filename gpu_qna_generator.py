import os
import json
import re
import random
import multiprocessing
from transformers import pipeline

# Choose a larger generative model
model_name = 'google/flan-t5-small'

# Load the pipeline (load once per process for efficiency)
def load_pipeline():
    return pipeline("text2text-generation", model=model_name, device=-1)

def generate_qa_pairs(input_file, output_dir, qna_generator):
    """
    Generates question-answer pairs from chunks in the input text file
    using a single generative model. Handles multiline text and ensures JSON output.

    Args:
        input_file (str): Path to the input text file.
        output_dir (str): Path to the output directory.
        qna_generator (pipeline): The loaded text generation pipeline.
    """
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.jsonl")
    print(f"[Process {multiprocessing.current_process().pid}] Processing file: {input_file}")

    qna_generation_prompts = [
        "Generate a question and its answer based on the following text. Format your response as 'Question: [your question] Answer: [your answer]'. Text:",
        "Create a question-answer pair from this text. Ensure the output follows this format: 'Question: [your question] Answer: [your answer]'. Text:",
        "Based on the text, ask a question and provide the answer, strictly adhering to the format 'Question: [your question] Answer: [your answer]'. Text:",
        "Formulate a question that the following text answers, and then provide the answer in the format 'Question: [your question] Answer: [your answer]'. Text:",
        "Generate a relevant question and answer from the text, presented as 'Question: [your question] Answer: [your answer]'. Text:"
    ]

    try:
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:
            chunk = ""
            chunk_number = 0
            for line in infile:
                line = line.strip()
                if line.startswith(f"{chunk_number + 1}."):
                    if chunk:
                        print(f"[Process {multiprocessing.current_process().pid}] Processing chunk {chunk_number}...")
                        generation_prompt = random.choice(qna_generation_prompts)
                        full_prompt = f"{generation_prompt} {chunk}"
                        try:
                            generated_output = qna_generator(full_prompt, max_length=256, num_return_sequences=1, temperature=0.8, do_sample=True)
                            print(generated_output)
                            if generated_output and generated_output[0] and 'generated_text' in generated_output[0]:
                                qa_pair = generated_output[0]['generated_text'].strip()
                                if "Question:" in qa_pair and "Answer:" in qa_pair:
                                    try:
                                        qa_match = re.search(r'Question:\s*(.*?)\s*Answer:\s*(.*)', chunk, re.DOTALL)

                                        if qa_match:
                                            question = qa_match.group(1).strip()
                                            answer = qa_match.group(2).strip()

                                            if question and answer:
                                                messages_data = {
                                                    "messages": [
                                                        {"role": "user", "content": question},
                                                        {"role": "assistant", "content": answer}
                                                    ]
                                                }
                                                json.dump(messages_data, outfile, ensure_ascii=False)
                                                outfile.write("\n")
                                            else:
                                                print(f"[Process {multiprocessing.current_process().pid}] Could not properly parse question and answer from: '{qa_pair}' in chunk {chunk_number} of {input_file}")
                                    except ValueError:
                                        print(f"[Process {multiprocessing.current_process().pid}] Could not split question and answer from: '{qa_pair}' in chunk {chunk_number} of {input_file}")
                                else:
                                    print(f"[Process {multiprocessing.current_process().pid}] Generated output does not contain 'Question:' and 'Answer:' in chunk {chunk_number} of {input_file}: '{qa_pair}'")
                            else:
                                print(f"[Process {multiprocessing.current_process().pid}] Could not generate question-answer pair for chunk {chunk_number} in {input_file}")

                        except Exception as e:
                            print(f"[Process {multiprocessing.current_process().pid}] Error generating question-answer pair for chunk {chunk_number} in {input_file}: {e}")

                    chunk = line[len(str(chunk_number + 1)) + 1:].strip()
                    chunk_number += 1
                elif line and chunk:
                    chunk += "\n" + line  # Handle multiline by joining with newline

            # Process the last chunk if any
            if chunk:
                print(f"[Process {multiprocessing.current_process().pid}] Processing chunk {chunk_number}...")
                generation_prompt = random.choice(qna_generation_prompts)
                full_prompt = f"{generation_prompt} {chunk}"
                try:
                    generated_output = qna_generator(full_prompt, max_length=256, num_return_sequences=1, temperature=0.8, do_sample=True)
                    print(generated_output)
                    if generated_output and generated_output[0] and 'generated_text' in generated_output[0]:
                        qa_pair = generated_output[0]['generated_text'].strip()
                        if "Question:" in qa_pair and "Answer:" in qa_pair:
                            try:
                                qa_match = re.search(r'Question:\s*(.*?)\s*Answer:\s*(.*)', chunk, re.DOTALL)

                                if qa_match:
                                    question = qa_match.group(1).strip()
                                    answer = qa_match.group(2).strip()
                                #question_part, answer_part = qa_pair.split("Answer:")
                                #question = question_part.split("Question:")[-1].strip()
                                #answer = answer_part.strip()
                                    if question and answer:
                                        messages_data = {
                                            "messages": [
                                                {"role": "user", "content": question},
                                                {"role": "assistant", "content": answer}
                                            ]
                                        }
                                        json.dump(messages_data, outfile, ensure_ascii=False)
                                        outfile.write("\n")
                                    else:
                                        print(f"[Process {multiprocessing.current_process().pid}] Could not properly parse question and answer from last chunk of {input_file}: '{qa_pair}'")
                            except ValueError:
                                print(f"[Process {multiprocessing.current_process().pid}] Could not split question and answer from last chunk of {input_file}: '{qa_pair}'")
                        else:
                            print(f"[Process {multiprocessing.current_process().pid}] Generated output does not contain 'Question:' and 'Answer:' in last chunk of {input_file}: '{qa_pair}'")
                    else:
                        print(f"[Process {multiprocessing.current_process().pid}] Could not generate question-answer pair for last chunk in {input_file}")

                except Exception as e:
                    print(f"[Process {multiprocessing.current_process().pid}] Error generating question-answer pair for last chunk in {input_file}: {e}")

        print(f"[Process {multiprocessing.current_process().pid}] QA pairs saved to: {output_file}")

    except FileNotFoundError:
        print(f"[Process {multiprocessing.current_process().pid}] Error: Input file not found: {input_file}")
    except Exception as e:
        print(f"[Process {multiprocessing.current_process().pid}] An error occurred while processing {input_file}: {e}")

def process_file(file_path, output_dir):
    """
    Worker function for multiprocessing. Processes a single file.
    """
    qna_generator = load_pipeline()
    generate_qa_pairs(file_path, output_dir, qna_generator)

def process_directory(directory, output_dir, num_processes):
    """
    Processes all text files in the given directory using multiple processes.

    Args:
        directory (str): Path to the directory containing the text files.
        output_dir (str): Path to the output directory.
        num_processes (int): Number of processes to use.
    """
    files_to_process = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".txt")]

    if files_to_process:
        print(f"Starting processing with {num_processes} processes...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = [(file_path, output_dir) for file_path in files_to_process]
            pool.starmap(process_file, tasks)
        print("Processing complete.")
    else:
        print("No .txt files found in the specified directory.")

if __name__ == "__main__":
    input_path = "data_refined"
    output_directory = "finetuning"
    os.makedirs(output_directory, exist_ok=True)

    num_cores = 12  # Use the number of cores you have
    num_processes_to_use = min(multiprocessing.cpu_count(), num_cores) # Limit to available cores

    if os.path.isdir(input_path):
        process_directory(input_path, output_directory, num_processes_to_use)
    elif os.path.isfile(input_path) and input_path.endswith(".txt"):
        process_file(input_path, output_directory)
    else:
        print("Invalid input path.")
