import json
import random
import networkx as nx
from tqdm import tqdm
from datasets import load_dataset


def remove_20_percent(lst):
    if len(lst) == 0:
        return lst
    random.seed(42)
    num_to_remove = max(1, int(len(lst) * 0.2))
    indices_to_remove = random.sample(range(len(lst)), num_to_remove)
    for i in sorted(indices_to_remove, reverse=True):
        del lst[i]
    return lst


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    graph = dict()
    for data in qa_dataset:
        qid = data["id"]
        graph_data = data["graph"]
        graph[qid] = {
            "graph_1": remove_20_percent(graph_data[:]),
            "graph_2": graph_data
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["graph_1"] = []
        sample["graph_2"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        sample["graph_1"] = graph[qid]["graph_1"]
        sample["graph_2"] = graph[qid]["graph_2"]
        if "graph" in sample:
            del sample["graph"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc)
    return qa_dataset


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_graph(graph: list) -> nx.Graph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.strip(), t.strip(), relation=r.strip())
    return G


def main():
    input_file = "./webqsp"
    dataset = load_dataset(input_file, split="test[:100]")
    rule_path = "results/gen_rule_path/webqsp/mkg4llm/test[:100]/predictions_3_False.jsonl"
    rule_dataset = load_jsonl(rule_path)
    update_dataset = merge_rule_result(dataset, rule_dataset)

    with open('multi_webqsp/test_dataset.jsonl', 'w', encoding='utf-8') as f:
        for sample in tqdm(update_dataset):
            sample_json = json.dumps(sample, ensure_ascii=False)
            f.write(sample_json + '\n')


if __name__ == "__main__":
    main()
