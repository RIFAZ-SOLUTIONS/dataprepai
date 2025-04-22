import os
import json
import random
import multiprocessing
from transformers import pipeline

# Choose models
question_generation_model_name = 'google/flan-t5-small'
answer_generation_model_name = 'google/flan-t5-xl'

# Load the pipelines (load once per process for efficiency)
def load_pipelines():
    question_generator = pipeline("text2text-generation", model=question_generation_model_name, device=-1)
    answer_generator = pipeline("text2text-generation", model=answer_generation_model_name, device=-1)
    return question_generator, answer_generator

def generate_qa_pairs(input_file, output_dir, question_generator, answer_generator):
    """
    Generates question-answer pairs using separate models for question and answer.

    Args:
        input_file (str): Path to the input text file.
        output_dir (str): Path to the output directory.
        question_generator (pipeline): The loaded question generation pipeline.
        answer_generator (pipeline): The loaded answer generation pipeline.
    """
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.jsonl")
    print(f"[Process {multiprocessing.current_process().pid}] Processing file: {input_file}")

    question_generation_prompts = [
        "Generate a question that can be answered by the following text:",
        "What is a relevant question for this text:",
        "Ask a question about:",
        "Formulate a question based on:",
        "Create a question that this text answers:"
    ]

    try:
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:
            chunk = ""
            chunk_number = 0
            for line in infile:
                line = line.strip()
                if line.startswith(f"{chunk_number + 1}ps."):
                    if chunk:
                        print(f"[Process {multiprocessing.current_process().pid}] Processing chunk {chunk_number}...")
                        question_generation_prompt = random.choice(question_generation_prompts)
                        try:
                            generated_questions = question_generator(f"{question_generation_prompt} {chunk}", max_length=64, num_return_sequences=1)
                            print(generated_questions)
                            if generated_questions and generated_questions[0] and 'generated_text' in generated_questions[0]:
                                question = generated_questions[0]['generated_text'].strip()
                                print(generated_questions)

                                # Now, use the answer generator with the question and the original chunk
                                answer_prompt = f"Answer the following question based on the provided text: Question: {question} Text: {chunk} Answer:"
                                try:
                                    generated_answers = answer_generator(answer_prompt, max_length=128, num_return_sequences=1, temperature=0.7, do_sample=True)
                                    print(generated_answers)
                                    if generated_answers and generated_answers[0] and 'generated_text' in generated_answers[0]:
                                        answer = generated_answers[0]['generated_text'].strip()
                                        
                                        messages_data = {
                                            "messages": [
                                                {"role": "user", "content":f"{chunk}\n {question}"},
                                                {"role": "assistant", "content": answer}
                                            ]
                                        }
                                        json.dump(messages_data, outfile, ensure_ascii=False)
                                        outfile.write("\n")
                                    else:
                                        print(f"[Process {multiprocessing.current_process().pid}] Could not generate answer for question: '{question}' in chunk {chunk_number} of {input_file}")

                                except Exception as e:
                                    print(f"[Process {multiprocessing.current_process().pid}] Error generating answer for chunk {chunk_number} in {input_file}: {e}")

                            else:
                                print(f"[Process {multiprocessing.current_process().pid}] Could not generate question for chunk {chunk_number} in {input_file}")

                        except Exception as e:
                            print(f"[Process {multiprocessing.current_process().pid}] Error generating question for chunk {chunk_number} in {input_file}: {e}")

                    chunk = line[len(str(chunk_number + 1)) + 1:].strip()
                    chunk_number += 1
                elif line and chunk:
                    chunk += "\n" + line  # Handle multiline by joining with newline

            # Process the last chunk if any
            if chunk:
                print(f"[Process {multiprocessing.current_process().pid}] Processing chunk {chunk_number}...")
                question_generation_prompt = random.choice(question_generation_prompts)
                try:
                    generated_questions = question_generator(f"{question_generation_prompt} {chunk}", max_length=64, num_return_sequences=1)
                    print(generated_questions)
                    if generated_questions and generated_questions[0] and 'generated_text' in generated_questions[0]:
                        question = generated_questions[0]['generated_text'].strip()

                        answer_prompt = f"Answer the following question based on the provided text: Question: {question} Text: {chunk} Answer:"
                        try:
                            generated_answers = answer_generator(answer_prompt, max_length=128, num_return_sequences=1, temperature=0.7, do_sample=True)
                            print(generated_answers)
                            if generated_answers and generated_answers[0] and 'generated_text' in generated_answers[0]:
                                answer = generated_answers[0]['generated_text'].strip()
                                messages_data = {
                                    "messages": [
                                        {"role": "user", "content": question},
                                        {"role": "assistant", "content": answer}
                                    ]
                                }
                                json.dump(messages_data, outfile, ensure_ascii=False)
                                outfile.write("\n")
                            else:
                                print(f"[Process {multiprocessing.current_process().pid}] Could not generate answer for question: '{question}' in last chunk of {input_file}")
                        except Exception as e:
                            print(f"[Process {multiprocessing.current_process().pid}] Error generating answer for last chunk in {input_file}: {e}")
                    else:
                        print(f"[Process {multiprocessing.current_process().pid}] Could not generate question for last chunk in {input_file}")

                except Exception as e:
                    print(f"[Process {multiprocessing.current_process().pid}] Error generating question for last chunk in {input_file}: {e}")

        print(f"[Process {multiprocessing.current_process().pid}] QA pairs saved to: {output_file}")

    except FileNotFoundError:
        print(f"[Process {multiprocessing.current_process().pid}] Error: Input file not found: {input_file}")
    except Exception as e:
        print(f"[Process {multiprocessing.current_process().pid}] An error occurred while processing {input_file}: {e}")

def process_file(file_path, output_dir):
    """
    Worker function for multiprocessing. Processes a single file.
    """
    question_generator, answer_generator = load_pipelines()
    generate_qa_pairs(file_path, output_dir, question_generator, answer_generator)

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
    input_path = "data_refined_sentences"
    output_directory = "finetuning"
    os.makedirs(output_directory, exist_ok=True)

    num_cores = 18  # Use the number of cores you have
    num_processes_to_use = min(multiprocessing.cpu_count(), num_cores) # Limit to available cores

    if os.path.isdir(input_path):
        process_directory(input_path, output_directory, num_processes_to_use)
    elif os.path.isfile(input_path) and input_path.endswith(".txt"):
        process_file(input_path, output_directory)
    else:
        print("Invalid input path.")
