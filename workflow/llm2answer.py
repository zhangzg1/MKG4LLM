import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
from tqdm import tqdm
from src.llms import get_registed_model
import os
from datasets import load_dataset
from src.utils.evaluate_results import eval_result
import json


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def prediction(data, processed_list, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None
    input_prompt = f"question: {question}\nPlease answer the given question and keep the answer as simple as possible and return all the possible answers as a list."
    prediction = model.generate_sentence(input_prompt)
    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input_prompt,
    }
    return result


def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)
    dataset = load_dataset(input_file, split=args.split)
    print("Load dataset from finished")

    output_dir = os.path.join(args.output_path, args.d, args.split, args.model_name)
    print("Save results to: ", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = LLM(args)

    print("Prepare pipline for inference...")
    model.prepare_for_inference()

    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    for data in tqdm(dataset):
        res = prediction(data, processed_list, model)
        if res is not None:
            fout.write(json.dumps(res) + "\n")
            fout.flush()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="data")
    argparser.add_argument("--d", "-d", type=str, default="webqsp")
    argparser.add_argument("--split", type=str, default="test[:100]")
    argparser.add_argument("--output_path", type=str, default="results/KGQA/no_kg_results")
    argparser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    argparser.add_argument("--force", "-f", action="store_true", help="force to overwrite the results")

    args, _ = argparser.parse_known_args()

    LLM = get_registed_model(args.model_name)
    LLM.add_args(argparser)
    args = argparser.parse_args()

    main(args, LLM)
