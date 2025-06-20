import transformers
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer, AutoTokenizer


class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path",
                            default='meta-llama/Llama-3.1-8B-Instruct')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='bf16')

    def __init__(self, args):
        self.args = args
        self.maximum_token = 4096 - 100

    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(**kwargs)
        return model

    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator = transformers.pipeline("text-generation", model=self.args.model_path, device_map="auto",
                                               model_kwargs=model_kwargs, tokenizer=self.tokenizer,
                                               torch_dtype=self.DTYPE.get(self.args.dtype, None))

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        messages = [{"role": "user", "content": llm_input}]
        terminators = [
            self.generator.tokenizer.eos_token_id,
            self.generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.generator(messages, max_new_tokens=self.args.max_new_tokens, eos_token_id=terminators)
        return outputs[0]["generated_text"][-1]["content"]
