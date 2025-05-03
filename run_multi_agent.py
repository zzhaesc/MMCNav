import json
import os
import argparse

import tqdm

from vln.dataset import load_dataset
from vln.prompt_builder import get_observations_lines_noimage
from vln.mmc_env import mmcEnv, get_gold_nav
from vln.mmc_agent import Planner, Executor, Reflector, Observer
from vln.evaluate import get_metrics_from_results
from llm.query_llm import OpenAI_LLM, LLaVA_OV

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--datasets_dir', default='./datasets', type=str)

# parser.add_argument('--dataset_name', default='map2seq', type=str)
parser.add_argument('--dataset_name', default='touchdown', type=str)

parser.add_argument('--split', default='dev', type=str)
parser.add_argument('--scenario', default='unseen', type=str)

parser.add_argument('--map_dir', default='Your map directory', type=str)
parser.add_argument('--image_dir', default='Your image directory', type=str)
parser.add_argument('--image', default='llava-OV', type=str)
parser.add_argument('--observation_file', default='observation_llava-OV.json', type=str)
parser.add_argument('--landmarks_name', default='gpt3_5shot', type=str)

parser.add_argument('--visual_model_path', default="LLaVA directory", type=str)

parser.add_argument('--model_name', default="gpt-4o", type=str)
parser.add_argument('--api_key', default='Your openai key')
parser.add_argument('--num_instances', default=-1, type=int)  # -1 for all instances
parser.add_argument('--max_tokens', default=512, type=int)

parser.add_argument('--temperature', default=0, type=int)

parser.add_argument('--max_steps', default=50, type=int)  # maximum number of agent steps before run is canceled

parser.add_argument('--output_dir', default='./outputs', type=str)
parser.add_argument('--seed', default=1, type=int)

opts = parser.parse_args()
API_key = opts.api_key
dataset_name = opts.dataset_name
is_map2seq = dataset_name == 'map2seq'
print("is_map2seq " + str(is_map2seq))
data_dir = opts.datasets_dir   # ./datasets
dataset_dir = os.path.join(data_dir, dataset_name + '_' + opts.scenario) 
graph_dir = os.path.join(dataset_dir, 'graph')  
map_dir = os.path.join(opts.map_dir, dataset_name) 
landmarks_dir = os.path.join(data_dir, 'landmarks')  
landmarks_file = os.path.join(landmarks_dir, dataset_name, f'{opts.landmarks_name}_unfiltered.json') 
prompts_dir = os.path.join('llm', 'prompts') 
image_dir = opts.image_dir
caption_sets_dir = os.path.join(data_dir, 'observation')

counter = 0

visual_model_name = opts.visual_model_path


def main():
    output_name = '_'.join(["MMCNav", opts.image])
    output_dir = os.path.join(opts.output_dir, dataset_name + "_" + opts.scenario,
                              f"{opts.model_name}")
    results_file = os.path.join(output_dir, f'{output_name}_{opts.split}.json')
    print(results_file)
    llm = OpenAI_LLM(max_tokens=opts.max_tokens,
                     model_name=opts.model_name,
                     api_key=API_key,
                     cache_name='navigation',
                     finish_reasons=['stop', 'length'])
    
    obs_llm = LLaVA_OV(visual_model_name, "llava_qwen")
    
    planner = Planner(llm, prompts_dir, "planner", data_dir, is_map2seq)
    observer = Observer(obs_llm, prompts_dir, image_dir, "observer", is_map2seq)
    executor = Executor(llm, prompts_dir, "executor", is_map2seq)
    reflector = Reflector(llm, prompts_dir, "reflector", is_map2seq)

    env = mmcEnv(planner, executor, reflector, observer,
                 is_map2seq, graph_dir, image_dir, map_dir
                 )
    
    os.makedirs(output_dir, exist_ok=True)
    instances, caption_sets  = load_dataset(opts.split, env, dataset_dir, dataset_name,
                                        caption_sets_dir, landmarks_file)


    results = dict()
    results['opts'] = vars(opts)
    results['instances'] = dict()

    if opts.num_instances != -1:
        instances = instances[:opts.num_instances]
    print('instances: ', len(instances))

    pbar = tqdm.tqdm(total=len(instances), smoothing=0.1)

    for i, instance in enumerate(instances):
        print(i, 'number of instances processed')
        print('idx', instance['idx'])
        result, caption_sets = process_instance((instance, pbar), env, caption_sets) 
        results['instances'][result['idx']] = result

        with open(os.path.join(caption_sets_dir, opts.observation_file), 'w') as f:
            json.dump(caption_sets, f, indent=2)

    correct, tc, spd, kpa, nDTW, results = get_metrics_from_results(results, env.graph)
    print('')
    print('correct', correct)
    print('tc', tc)
    print('spd', spd)
    print('kpa', kpa)
    print('nDTW', nDTW)
    print('')

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote results to: ', results_file)


def process_instance(args, env, caption):
    instance, pbar = args

    nav, navigation_lines, is_actions, subtask_judge, navigation_task, caption = env.run(opts.max_steps, instance, caption)

    gold_nav = get_gold_nav(instance, env.graph)
    gold_navigation_lines, gold_is_actions = get_observations_lines_noimage(gold_nav, env.get_observations_noimage)

    global counter
    counter += 1
    pbar.update()

    print('instance id', instance["id"])
    print('result:')
    print(instance['navigation_text'])
    print(instance['landmarks'])
    print('\n'.join(navigation_lines))
    print('actions', nav.actions)
    print('processed instances', counter)
    print("=====================================================")
    result = dict(idx=instance['idx'],
                  navigation_text=instance['navigation_text'],
                  navigation_task=navigation_task,
                  start_heading=instance['start_heading'],
                  gold_actions=gold_nav.actions,
                  gold_states=gold_nav.states,
                  gold_pano_path=instance['route_panoids'],
                  gold_navigation_lines=gold_navigation_lines,
                  gold_is_actions=gold_is_actions,
                  agent_actions=nav.actions,
                  agent_states=nav.states,
                  agent_pano_path=nav.pano_path,
                  agent_navigation_lines=navigation_lines,
                  agent_is_actions=is_actions,
                  subtask_judge=subtask_judge
                  )
    
    return result, caption

if __name__ == '__main__':
    main()