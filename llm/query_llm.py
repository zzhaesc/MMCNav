import json
import os
import time

import fire
import openai
import json
import base64
from openai import OpenAI

import copy
import torch
from PIL import Image

import re
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import process_images, tokenizer_image_token
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from LLaVA.llava.conversation import conv_templates



class LLM:
    def __init__(self, api_key, model_name, max_tokens, cache_name='default', **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.queried_tokens = 0

        cache_model_dir = os.path.join('llm', 'cache', self.model_name)
        os.makedirs(cache_model_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_model_dir, f'{cache_name}.json')
        self.cache = dict()

        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def query_api(self, prompt):
        raise NotImplementedError

    def get_cache(self, prompt, instance_idx):
        sequences = self.cache.get(instance_idx, [])

        for sequence in sequences:
            if sequence.startswith(prompt) and len(sequence) > len(prompt)+1:
                return sequence
        return None

    def add_to_cache(self, sequence, instance_idx):
        if instance_idx not in self.cache:
            self.cache[instance_idx] = []
        sequences = self.cache[instance_idx]

        # newest result to the front
        sequences.append(sequence)

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print('cache saved to: ' + self.cache_file)

    def get_sequence(self, prompt, instance_idx, read_cache=True):
        sequence = None
        if read_cache:
            sequence = self.get_cache(prompt, instance_idx)
        print('cached sequence')
        if sequence is None:
            print('query API')
            sequence = self.query_api(prompt)
            self.add_to_cache(sequence, instance_idx)
            #print('api sequence')
        return sequence


class OpenAI_LLM(LLM):

    def __init__(self, model_name, api_key, logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):

        self.logit_bias = logit_bias
        
        self.finish_reasons = finish_reasons

        self.client = OpenAI(
            api_key=api_key
            )

        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        self.INITIAL_WAIT_TIME = 1
        self.MAX_RETRIES = 10
        super().__init__(api_key, model_name, max_tokens, **kwargs)


    def query(self, prompt, map_path=None):
        retries = 0
        wait_time = self.INITIAL_WAIT_TIME

        while retries < self.MAX_RETRIES:
            try:
                if map_path:
                    with open(map_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text", 
                                        "text": f"{prompt}"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url":{
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "low"
                                        }
                                    }
                                ],
                            }
                        ],
                        max_tokens=self.max_tokens,
                        temperature=0.
                    )
                else:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text", 
                                        "text": f"{prompt}"
                                    },
                                ],
                            }
                        ],
                        temperature=0.
                        )
                if response is not None:
                    if response.choices[0].message.content is not None:
                        return response.choices[0].message.content.strip(), response.usage.total_tokens
                else:
                    print("Incomplete response received. Retry!")
                    time.sleep(wait_time)
                    wait_time *= 2
                    retries += 1
            except Exception as e:
                print("Error occurred: {}".format(e))
                print("Retrying gpt after {} seconds...".format(wait_time))
                time.sleep(wait_time)
                if retries < 5:
                    wait_time *= 2
                retries += 1
        raise Exception("Exceeded maximum number of retries. Could not complete the request.")

class LLaVA_OV:
    def __init__(self, model_path, model_name, device="cuda" , device_map="auto"):
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(model_path, 
                                                                      None, model_name, 
                                                                      device_map=device_map)  # Add any other thing you want to pass in llava_model_args
        self.model = self.model.eval()
        self.device = device


    def predict(self, prompt, image_path=None, image=None):
        conv_template = "qwen_1_5"
        if image_path or image:
            if image_path:
                image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

            question = DEFAULT_IMAGE_TOKEN + prompt
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            image_sizes = [image.size]

            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            return text_outputs
        else:
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)

            cont = self.model.generate(
                    input_ids,
                    # images=image_tensor,
                    # image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
            )
            summarize = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            return summarize

    def traffic_flow_judge(self, prompt, front_rear_image):
        conv_template = "qwen_1_5"
        image_tensors = process_images(front_rear_image, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
        question = DEFAULT_IMAGE_TOKEN + " " + DEFAULT_IMAGE_TOKEN + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [img.size for img in front_rear_image]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        
        return text_outputs
