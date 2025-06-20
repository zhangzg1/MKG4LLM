import random
import time
import re
import string
import networkx as nx
from collections import deque
from typing import Callable
from datetime import timedelta


def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


def path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
    return result.strip()


def bfs_with_rule(graph, start_node, target_rule):
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))
    return result_paths


class PromptBuilder(object):
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, each path is prefixed with a weight value, and the range of the weight values is from 0.0 to 1.0\
When addressing the problem, it is crucial to consider all reasoning paths comprehensively and not to focus solely on those with higher weights, \
as paths with lower weights are equally significant. The magnitude of the weight merely indicates the level of importance of the reasoning path. \
Please answer the given question, and keep the answer as simple as possible and return all the possible answers as a list."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    EACH_LINE = """ Please return each answer in a new line."""

    def __init__(self, prompt_path, cot=False, explain=False, each_line=False, maximum_token=4096,
                 tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.cot = cot
        self.explain = explain
        self.each_line = each_line
        self.maximum_token = maximum_token
        self.tokenize = tokenize

    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template

    def apply_rules(self, graph, rules, source_entities):
        results = []
        for entity in source_entities:
            for rule in rules:
                res = bfs_with_rule(graph, entity, rule)
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

    def process_input(self, question_dict, w1, w2):
        question = question_dict['question']

        if not question.endswith('?'):
            question += '?'

        list_paths = dict()
        graph_1 = build_graph(question_dict['graph_1'])
        graph_2 = build_graph(question_dict['graph_2'])
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        if len(rules) > 0:
            reasoning_paths_1 = self.apply_rules(graph_1, rules, entities)
            reasoning_paths_2 = self.apply_rules(graph_2, rules, entities)
            lists_of_paths_1 = [path_to_string(p) for p in reasoning_paths_1]
            lists_of_paths_2 = [path_to_string(p) for p in reasoning_paths_2]
            list_paths[w1] = lists_of_paths_1
            list_paths[w2] = lists_of_paths_2

        input = self.QUESTION.format(question=question)
        instruction = self.SAQ_RULE_INSTRUCTION

        if self.cot:
            instruction += self.COT

        if self.explain:
            instruction += self.EXPLAIN

        if self.each_line:
            instruction += self.EACH_LINE

        other_prompt = self.prompt_template.format(instruction=instruction,
                                                   input=self.GRAPH_CONTEXT.format(context="") + input)
        pre_context = self.check_prompt_length(other_prompt, list_paths, self.maximum_token)
        context = self.merge_reasoning_paths(pre_context)
        input = self.GRAPH_CONTEXT.format(context=context) + input
        input = self.prompt_template.format(instruction=instruction, input=input)
        return input

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


def init_model(args, LLM):
    model = LLM(args)
    input_builder = PromptBuilder(
        args.prompt_path,
        cot=args.cot,
        explain=args.explain,
        each_line=args.each_line,
        maximum_token=model.maximum_token,
        tokenize=model.tokenize
    )
    model.prepare_for_inference()
    return input_builder, model


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_dataset(results):
    acc_list = []
    for result in results:
        prediction = result['prediction']
        answer = result['ground_truth']
        acc = eval_acc(prediction, answer)
        acc_list.append(acc)
    return sum(acc_list) / len(acc_list)
