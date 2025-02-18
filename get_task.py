import json
import os
import argparse
import time
import random

import tqdm

from vln.dataset import load_dataset, get_planform_map
from vln.prompt_builder import get_navigation_lines

from vln.env import ClipEnv, get_gold_nav

from vln.evaluate import get_metrics_from_results
from vln.agent import Agent
from openai import OpenAI

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='touchdown', type=str)
# parser.add_argument('--dataset_name', default='map2seq', type=str)
parser.add_argument('--baseline', default='forward', type=str, choices=['forward', 'random'])
parser.add_argument('--split', default='dev', type=str)
parser.add_argument('--scenario', default='unseen', type=str)
parser.add_argument('--num_instances', default=-1, type=int)  # -1 for all instances

parser.add_argument('--landmarks_name', default='gpt3_5shot', choices=['gpt3_0shot', 'gpt3_5shot'], type=str)

parser.add_argument('--seed', default=1, type=int)
opts = parser.parse_args()

random.seed(opts.seed)

split = opts.split
num_instances = opts.num_instances

dataset_name = opts.dataset_name
scenario = opts.scenario
is_map2seq = dataset_name == 'map2seq'

data_dir = opts.datasets_dir
dataset_dir = os.path.join(data_dir, dataset_name + '_' + scenario)
graph_dir = os.path.join(dataset_dir, 'graph')
planform_map_dir = os.path.join(data_dir, 'planform')
landmarks_dir = os.path.join(data_dir, 'landmarks')
landmarks_file = os.path.join(landmarks_dir, opts.dataset_name, f'{opts.landmarks_name}_unfiltered.json')
prompts_dir = os.path.join('llm', 'prompts', 'instructionPlan')
counter = 0
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def main():
    env = ClipEnv(graph_dir, panoCLIP=None)

    instances = load_dataset(split, env, dataset_dir, dataset_name, landmarks_file)

    client = OpenAI(
            base_url="https://mtu.mtuopenai.xyz/v1",
            api_key='sk-FastAPI1FSi0qpFC1wGtf3Jjl0lA1csUm0kEl2XIp3l1BHbm'
            )

    if num_instances != -1:
        instances = instances[:num_instances]
    print('instances: ', len(instances))

    with open(os.path.join(prompts_dir, 'subtask_split.txt')) as f: 
        prompt_template = ''.join(f.readlines())
    
    results = dict()
    results['instances'] = dict()
    for i, instance in tqdm.tqdm(list(enumerate(instances))):
        print(i, 'number of instances processed')
        print('idx', instance['idx'])
        # if instance['idx'] == '7277':
        if i>=520 and i<=539:
            prompt = prompt_template.format(instance['navigation_text'])
            response = client.chat.completions.create(
                model='gpt-4o-2024-08-06',
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
                # max_tokens=300,
                temperature=0.
            )
            # print(response.choices[0].message.content)
            result = dict(
                idx=instance['idx'],
                navigation_text=instance['navigation_text'],
                navigation_task=response.choices[0].message.content
            )
            results['instances'][result['idx']] = result

    results_file = os.path.join(data_dir, "subtask", dataset_name, "get_subtask_dev_test.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote results to: ', results_file)
    



if __name__ == '__main__':
    main()
