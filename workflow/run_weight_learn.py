import time
import torch
import torch.nn as nn
import argparse
from src.utils.weight_train_eval import train
from src.utils.load_data import build_iterator, build_dataset
from src.utils.weight_train_utils import init_model, get_time_dif
from src.llms import get_registed_model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="dataset_path", default="data/multi_webqsp")
parser.add_argument("--model_name", type=str, help="model_name for save results", default="mkg4llm")
parser.add_argument("--prompt_path", type=str, help="prompt_path", default="src/prompts/llama_prompt.txt")
parser.add_argument("--cot", action="store_true", default=False)
parser.add_argument("--explain", action="store_true", default=False)
parser.add_argument("--each_line", action="store_true", default=False)
args, _ = parser.parse_known_args()


class Config(object):

    def __init__(self, dataset):
        self.train_path = dataset + '/train_dataset.jsonl'            # 训练集
        self.dev_path = dataset + '/validation_test.jsonl'            # 验证集
        self.output_path = dataset + '/saved_result/output.txt'       # 训练结果
        self.require_improvement = 12                                 # 若超过多少batch效果还没提升，则提前结束训练
        self.num_epochs = 3                                           # epoch数
        self.batch_size = 512                                         # mini-batch大小
        self.eval_steps = 3                                           # 经过多少步后进行评估
        self.learning_rate = 0.05                                     # 学习率


class WeightedParameters(nn.Module):
    def __init__(self):
        super(WeightedParameters, self).__init__()
        # self.a = nn.Parameter(torch.tensor([3.0, 1.0]))  # webqsp
        self.a = nn.Parameter(torch.tensor([2.5, 1.5]))    # cwq

    def forward(self):
        w = torch.softmax(self.a, dim=0)
        return w[0], w[1]


if __name__ == '__main__':
    config = Config(args.dataset_path)

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    time_dif = get_time_dif(start_time)

    print("Init Large language model...")
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()
    input_builder, llm_model = init_model(args, LLM)

    print("Start training...")
    model = WeightedParameters()
    train(config, model, llm_model, input_builder, train_iter, dev_iter)
