import json
import os
import requests

from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def get_token_limit(model='gpt-3.5-turbo'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit


class ChatGPT(BaseLanguageModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_gpt', choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-3.5-turbo')

    def __init__(self, args):
        self.args = args
        self.maximum_token = get_token_limit(self.args.model_gpt)
        self.redundant_tokens = 150

    def tokenize(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.args.model_gpt)
            num_tokens = len(encoding.encode(text))
        except KeyError:
            raise KeyError(f"Warning: model {self.args.model_gpt} not found.")
        return num_tokens + self.redundant_tokens

    def prepare_for_inference(self, model_kwargs={}):
        '''
        ChatGPT model does not need to prepare for inference
        '''
        pass

    def get_response(self, prompt: str):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        }
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.args.model_gpt, "messages": messages}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            res = response.json()
            return res['choices'][0]['message']['content']
        except requests.exceptions.RequestException as err:
            print("Request error occurred:", err)
            return None

    def generate_sentence(self, llm_input):
        input_length = self.tokenize(llm_input)
        if input_length > self.maximum_token:
            llm_input = llm_input[:self.maximum_token]
        response = self.get_response(llm_input)
        return response
