import json
import os
import argparse
import time
import random

from tqdm import tqdm

from vln.dataset import load_dataset, get_planform_map
from vln.prompt_builder import get_navigation_lines

from vln.env import ClipEnv, get_gold_nav
from vln.landmarks import filter_landmarks_5shot
from vln.evaluate import get_metrics_from_results
from vln.agent import Agent
from openai import OpenAI

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--datasets_dir', default='./datasets', type=str)
# parser.add_argument('--dataset_name', default='map2seq', type=str)
parser.add_argument('--dataset_name', default='touchdown', type=str)
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
    client = OpenAI(
            base_url="https://mtu.mtuopenai.xyz/v1",
            api_key='sk-FastAPI1FSi0qpFC1wGtf3Jjl0lA1csUm0kEl2XIp3l1BHbm'
            )
    results = dict()
    results['instances'] = dict()

    with open(os.path.join(prompts_dir, 'subtask_turing.txt')) as f: 
        prompt_template = ''.join(f.readlines())
    instances = list()
    with open(landmarks_file) as f:
        org_landmarks = json.load(f)['instances']

    with open(os.path.join(data_dir, 'subtask', dataset_name, 'get_subtask_dev_test.json'), 'r', encoding='utf-8') as f:
        instances = json.load(f)['instances']
        instances = list(instances.items())
        # if num_instances != -1:
        #     instances = instances[:num_instances]
        print('instances: ', len(instances))
        pbar = tqdm(total=len(instances), smoothing=0.1)
        for i, instance in enumerate(instances):
            # if instance[0] == '5492':
                idx = instance[0]
                if idx not in org_landmarks:
                    unfiltered = []
                else:
                    unfiltered = org_landmarks[idx]['unfiltered']
                landmarks = filter_landmarks_5shot(unfiltered)
                landmarks = '\n'.join(landmarks)

                navigation_text = instance[1]['navigation_text']
                navigation_task = instance[1]['navigation_task']
                
                print(i, 'number of instances processed')
                # print('idx', idx)
                prompt = prompt_template.format(navigation_task, landmarks)
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
                    # max_tokens=600,
                    temperature=0.
                )
            # print(response.choices[0].message.content)
                result = dict(
                    idx=idx,
                    navigation_text=navigation_text,
                    navigation_task=response.choices[0].message.content
                )
                results['instances'][result['idx']] = result
                pbar.update()

    results_file = os.path.join(data_dir, "subtask", dataset_name, "dev_turing_test.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote results to: ', results_file)
    

if __name__ == '__main__':
    main()
