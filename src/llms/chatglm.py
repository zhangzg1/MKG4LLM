from transformers import AutoTokenizer, AutoModel
import torch
from .base_language_model import BaseLanguageModel


class ChatGLM(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path",
                            default='THUDM/chatglm3-6b')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, args):
        self.args = args
        # self.maximum_token = 2048 - 100
        self.maximum_token = 512 - 100

    def load_model(self, **kwargs):
        model = AutoModel.from_pretrained(**kwargs)
        return model

    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        self.generator = AutoModel.from_pretrained(self.args.model_path, trust_remote_code=True, device_map="auto",
                                                   torch_dtype=self.DTYPE.get(self.args.dtype, None))

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs, _ = self.generator.chat(self.tokenizer, llm_input,
                                         max_length=self.args.max_new_tokens + self.maximum_token)
        return outputs
