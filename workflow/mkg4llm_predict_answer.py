import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from src.utils import utils
import argparse
from tqdm import tqdm
from src.llms import get_registed_model
import os
from src.utils.evaluate_results import eval_result
import json
from multiprocessing import Pool
from src.utils.build_qa_input import PromptBuilder
from functools import partial


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


def prediction(data, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }
    input = input_builder.process_input(data)
    prediction = model.generate_sentence(input)
    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input,
    }
    return result


def main(args, LLM):
    dataset_path = os.path.join(args.data_path, args.d)
    dataset = utils.load_jsonl(dataset_path)

    rule_postfix = "predict_result"
    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.each_line:
        rule_postfix += "_each_line"

    print("Load dataset from finished")
    output_dir = os.path.join(args.predict_path, args.model_name, rule_postfix)
    print("Save results to: ", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = LLM(args)
    input_builder = PromptBuilder(
        args.prompt_path,
        cot=args.cot,
        explain=args.explain,
        each_line=args.each_line,
        maximum_token=model.maximum_token,
        tokenize=model.tokenize
    )
    print("Prepare pipline for inference...")
    model.prepare_for_inference()

    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                    p.imap(
                        partial(
                            prediction,
                            processed_list=processed_list,
                            input_builder=input_builder,
                            model=model,
                        ),
                        dataset,
                    ),
                    total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(data, processed_list, input_builder, model)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="data")
    argparser.add_argument("--d", "-d", type=str, default="multi_webqsp/test_dataset.jsonl")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA/multi_webqsp")
    argparser.add_argument("--model_name", type=str, default="mkg4llm")
    argparser.add_argument("--prompt_path", type=str, help="prompt_path",
                           default="src/prompts/llama_prompt.txt")
    argparser.add_argument("--rule_path", type=str,
                           default="results/gen_rule_path/webqsp/mkg4llm/test[:100]/predictions_3_False.jsonl")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument("--force", "-f", action="store_true", help="force to overwrite the results")
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--debug", action="store_true")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
