from tqdm import tqdm
import json
from src.utils import utils, graph_utils
import random
from typing import Callable
from transformers import AutoTokenizer


class PromptBuilder(object):
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, each path is prefixed with a weight value, and the range of the weight values is from 0.0 to 1.0.\
When addressing the problem, it is crucial to consider all reasoning paths comprehensively and not to focus solely on those with higher weights, \
as paths with lower weights are equally significant. The magnitude of the weight merely indicates the level of importance of the reasoning path. \
Please answer the given question, and keep the answer as simple as possible and return all the possible answers as a list."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""

    def __init__(self, prompt_path, add_rule=True, use_true=False, cot=False, explain=False, use_random=False,
                 each_line=False, maximum_token=4096-200, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximum_token = maximum_token
        self.tokenize = tokenize
        self.each_line = each_line

    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template

    def apply_rules(self, graph, rules, source_entities):
        results = []
        for entity in source_entities:
            for rule in rules:
                res = graph_utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results

    def merge_reasoning_paths(self, reasoning_paths):
        if reasoning_paths is None:
            return ""
        path_dict = {}
        for line in reasoning_paths.split('\n'):
            if line.strip():
                weight, path = line.split('   ', 1)
                weight = float(weight.strip())
                path = path.strip()
                if path in path_dict:
                    path_dict[path] += weight
                else:
                    path_dict[path] = weight

        merged_paths = []
        for path, weight in path_dict.items():
            merged_paths.append(f"{weight:.1f}   {path}")
        return "\n".join(merged_paths)

    def process_input(self, question_dict):
        question = question_dict['question']

        if not question.endswith('?'):
            question += '?'

        list_paths = dict()
        if self.add_rule:
            graph_1 = graph_utils.build_graph(question_dict['graph_1'])
            graph_2 = graph_utils.build_graph(question_dict['graph_2'])
            entities = question_dict['q_entity']
            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = graph_utils.get_random_paths(entities, graph_1)
            else:
                rules = question_dict['predicted_paths']
            if len(rules) > 0:
                # 根据关系路径获取推理路径
                reasoning_paths_1 = self.apply_rules(graph_1, rules, entities)
                reasoning_paths_2 = self.apply_rules(graph_2, rules, entities)
                lists_of_paths_1 = [utils.path_to_string(p) for p in reasoning_paths_1]
                lists_of_paths_2 = [utils.path_to_string(p) for p in reasoning_paths_2]
                list_paths['0.1'] = lists_of_paths_1
                list_paths['0.9'] = lists_of_paths_2

        input = self.QUESTION.format(question=question)

        instruction = self.SAQ_RULE_INSTRUCTION

        other_prompt = self.prompt_template.format(instruction=instruction,
                                                   input=self.GRAPH_CONTEXT.format(context="") + input)

        pre_context = self.check_prompt_length(other_prompt, list_paths, self.maximum_token)

        context = self.merge_reasoning_paths(pre_context)

        input = self.GRAPH_CONTEXT.format(context=context) + input

        return instruction, input

    def check_prompt_length(self, prompt, dict_of_paths, maximum_token):
        list_of_all_paths = []
        for score, paths in dict_of_paths.items():
            for p in paths:
                list_of_all_paths.append(f"{score}   {p}")

        all_paths = "\n".join(list_of_all_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximum_token:
            return all_paths
        else:
            random.shuffle(list_of_all_paths)
            new_list_of_all_paths = []
            for path in list_of_all_paths:
                tmp_all_paths = "\n".join(new_list_of_all_paths + [path])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximum_token:
                    return "\n".join(new_list_of_all_paths)
                new_list_of_all_paths.append(path)


def prediction(data, input_builder):
    answer = data["answer"]
    instruction, input = input_builder.process_input(data)

    return instruction, input, answer


def input_tokenize(text):
    tokenizer = AutoTokenizer.from_pretrained('model/RoG')
    return len(tokenizer.tokenize(text))


def test():
    dataset = utils.load_jsonl("data/multi_webqsp/test_dataset.jsonl")

    input_builder = PromptBuilder(
        "src/prompts/llama_prompt.txt",
        tokenize=input_tokenize
    )

    results = []
    for data in tqdm(dataset):
        instruction, input, output = prediction(data, input_builder)
        result = {
            "instruction": instruction,
            "input": input,
            "output": output
        }
        results.append(result)

    with open('src/llm_finetune/data/train_dataset.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    test()
