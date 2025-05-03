from vln.mmc_env import navState, get_nav_from_actions
from vln.prompt_builder import get_observations_lines, concat_caption

from vln.dataset import get_pano_slices
from vln.base_memory import Memory, mapMaker

import PIL.Image as Image

import os
import re
import json


class Agent:
    def __init__(self, llm, agent_role):
        self.llm = llm
        self.agent_role = agent_role
            
    def extract_result(self, predict):
        return predict


class Planner(Agent):
    def __init__(self, llm, prompts_dir, agent_role, data_dir, is_map2seq):
        self.data_dir = data_dir
        self.is_map2seq = is_map2seq
        self.total_tokens = 0
        self.prompts_dir = os.path.join(prompts_dir, 'instructionPlan')
        with open(os.path.join(self.prompts_dir, 'task_judge_map2seq.txt' if self.is_map2seq else 'task_judge_touchdown.txt')) as f:
            self.judge_prompt_template = ''.join(f.readlines())

        with open(os.path.join(self.prompts_dir, 'task_split.txt')) as f:
            self.split_task_prompt_template = ''.join(f.readline())
        
        with open(os.path.join(self.prompts_dir, 'task_standardizing.txt')) as f:
            self.standardization_prompt_template = ''.join(f.readline())
        super().__init__(llm, agent_role)

    def subtask_judge(self, sub_nav_lines, sub_nav_plan, curr_obs, caption_name, is_print=False, split=True, dispatch=True, cap=True):
        caption = concat_caption(curr_obs[caption_name]["caption"])
        prompt = self.judge_prompt_template.format(
                                            sub_nav_plan,
                                            sub_nav_lines,
                                            caption
                                            )
        if is_print:
            print("sub_task:\n", sub_nav_plan)
            print("sub_nav_lines:\n", sub_nav_lines)
            print(prompt)
        predict, usage = self.llm.query(prompt, map_path=None)
        self.total_tokens += usage
        is_complete = self.extract_result(predict)
        return is_complete

    def get_navigation_plan(self, instance, is_map2seq, fromjson=True):
        if fromjson:
            id = instance['id']
            with open(os.path.join(self.data_dir, "task", "map2seq" if is_map2seq else "touchdown", "dev.json")) as f:
                navigation_plan = json.load(f)['instances'][f'{id}']['navigation_task']
            return navigation_plan
        
        navigation_text = instance["navigation_text"]
        prompt = self.split_task_prompt_template.format(navigation_text)
        subtask, usage  = self.llm.query(prompt)
        self.total_tokens += usage
        navigation_plan = self.standardize_task(subtask, instance['landmarks'])
        return navigation_plan

    def standardize_task(self, subtask, landmarks):
        prompt = self.standardization_prompt_template.format(subtask, landmarks)

        navigation_plan, usage = self.llm.query(prompt)
        self.total_tokens += usage
        return navigation_plan

    def extract_result(self, predict):
        match = re.search(r'[\'"“”]?answer[\'"“”]?\s*:\s*[\'"“”]?([^\'"“”]+)[\'"“”]?',predict, re.IGNORECASE)
        match_reason = re.search(r'"?reason"?: "([^"]+)"', predict, re.IGNORECASE)
        if match:
            result = match.group(1)
            print("answer:", result)
            if match_reason:
                reason = match_reason.group(1)
            else:
                print("failed to extract 'reason':", predict)
            if 'yes' in result.lower():
                return 'yes'
            else:
                return 'no'
        else:
            print("Failed to extract 'answer' from response: {}".format(predict))
            print("Default no")
            return 'no'


class Executor(Agent):
    def __init__(self, llm, prompts_dir, agent_role, is_map2seq):
        self.is_map2seq = is_map2seq
        self.total_tokens = 0

        self.prompts_dir = os.path.join(prompts_dir, 'map2seq' if self.is_map2seq else 'touchdown', 'navigation')
        with open(os.path.join(self.prompts_dir, 'instruction.txt')) as f:
            self.init_prompt_template = ''.join(f.readlines())

        with open(os.path.join(self.prompts_dir, 'advice_instruction.txt')) as f:
            self.advice_prompt_template = ''.join(f.readlines())

        super().__init__(llm, agent_role)
        
    def init_nav(self, instance, graph):
        actions = ['init']
        if self.is_map2seq:
            actions.append('forward')
        nav = get_nav_from_actions(actions, instance, graph)
        return nav

    def copy_nav(self, actions, instance, graph):
        nav = get_nav_from_actions(actions, instance, graph)
        return nav

    def run_execute(self, max_steps, step_id, mem, current_sub_index, nav, caption, get_observation_func, subtask_judge_func, advice=None):
        sub_nav_lines = list()
        no_stop = True
        not_pass = True
        init_panoid, init_heading = nav.get_state_from_index(0)
        while step_id <= max_steps:
            if current_sub_index > len(mem.taskPools): 
                current_sub_index = current_sub_index - 1  
                action = 'stop'
                nav.step(action)
                mem.add_subtask_judge(True)
                no_stop = False
                return nav, mem, current_sub_index, step_id, no_stop, caption

            panoid, heading = nav.get_state()
            if not self.is_map2seq and init_panoid == panoid and init_heading == heading:
                caption_name = panoid + '_' + str(heading) + '_' + 'init'
            else:
                caption_name = panoid + '_' + str(heading)

            if not_pass:
                new_navigation_lines, new_is_actions, caption, summary = get_observations_lines(nav, mem.landmarks,
                                                                               get_observation_func,
                                                                               caption,
                                                                               step_id)
                mem.add_summary(summary)
                mem.add_global_nav_sequences([{"line": line, "sub_index": current_sub_index} for line in new_navigation_lines])
                mem.add_global_isActions_sequences([{"is_actions": act, "sub_index": current_sub_index} for act in new_is_actions])
                

                sub_nav_lines = '\n'.join(mem.get_subtask_lines(current_sub_index))

                if advice is None:
                    if not self.is_map2seq and step_id == 0:
                        pass
                    else:
                        curr_obs = next((item for item in caption if caption_name in item), None)
                        judge_result = subtask_judge_func(sub_nav_lines, mem.taskPools[current_sub_index], curr_obs, caption_name)
                        if judge_result == 'yes':
                            print("Current subtasks completed")
                            if current_sub_index == len(mem.taskPools):
                                print("last-subtask completed")
                                mem.add_subtask_judge(False)
                                step_id = len(nav.actions)
                                break
                            mem.add_subtask_judge(True)
                            current_sub_index += 1
                            sub_nav_lines = '\n'.join(mem.get_subtask_lines(current_sub_index))
                        else:
                            print("Current subtask not completed")
                            mem.add_subtask_judge(False)
                
            step_id = len(nav.actions)
            curr_obs = next((item for item in caption if caption_name in item), None)
            
            if advice is None:
                prompt = self.init_prompt_template.format(f'{mem.taskPools[current_sub_index]}', f'{sub_nav_lines}',
                                                           concat_caption(curr_obs[caption_name]["caption"]))
            else:
                print("expert advice")
                prompt = self.advice_prompt_template.format(f'{mem.taskPools[current_sub_index]}', f'{sub_nav_lines}',
                                                            concat_caption(curr_obs[caption_name]["caption"]), f'{advice}')
                advice = None
            if step_id > max_steps:
                action = 'stop'
            else:
                action = self.query_next_action(prompt)
           
            print("next action:")
            print(action)

            action, reach_edge = nav.validate_action(action)
            print('Validated action', action)

            not_pass = True
            if action == 'stop' and current_sub_index == len(mem.taskPools):  # Last task and action is stop
                nav.step(action)
                mem.add_subtask_judge(True)
                no_stop = False
                return nav, mem, current_sub_index, step_id, no_stop, caption
            elif action == 'stop' and current_sub_index != len(mem.taskPools):
                current_sub_index += 1
                not_pass = False
                print("Can't stop until the task is completed")
                sub_nav_lines = '\n'.join(mem.get_subtask_lines(current_sub_index))
                if len(mem.subtask_judge) == 0:
                    pass
                else:
                    mem.add_subtask_judge(True, -1)
            else:
                nav.step(action)

            if reach_edge:
                break
            print("=======end of this step==========")
        return nav, mem, current_sub_index, step_id, no_stop, caption

    def retry(self, org_nav, org_mem, refc_mem, org_current_sub_index, caption, max_steps, reflection_func, subtask_judge_func, get_observation_func, graph):
        no_stop = False
        error_point, error_action = None, ""

        for _ in range(5):
            if error_point is None and "stop" not in error_action:
                error_point, error_action, advice = reflection_func(org_mem.task_list, org_mem.get_global_nav_lines(), caption, org_nav, org_mem.landmarks_dic, org_mem.get_map_path())
            else:
                break
        
        if error_point is None:
            error_point = len(org_nav.actions) - 1
        step_id = len(org_nav.actions)
        not_continue = True
        if_update = False
        refc_nav = None

        if org_nav.actions[error_point] not in error_action.lower():

            error_action = "stop"
        if error_point:
            if error_point >= len(org_nav.actions):

                return org_nav, org_mem, org_current_sub_index, step_id, no_stop, not_continue, if_update, caption
        
        if "stop" in error_action.lower() or error_point == len(org_nav.actions) - 1:  # If it's STOP that's wrong, it needs to be rolled back once
            if step_id > max_steps:
                return org_nav, org_mem, org_current_sub_index, step_id, no_stop, not_continue, if_update, caption
            
            roll_actions = org_nav.actions[:-1]
            roll_subtask_judge = org_mem.get_subtask_judge(-1)
            assert len(roll_actions) - 1 == len(roll_subtask_judge)

            refc_nav, refc_mem, refc_current_sub_index, refc_step_id = self.roll_back(roll_actions, refc_mem,
                                                                                            roll_subtask_judge,
                                                                                            1, caption, max_steps, get_observation_func, graph)
            refc_nav, refc_mem, refc_current_sub_index, refc_step_id, no_stop, caption = self.run_execute(max_steps, refc_step_id,
                                                                                    refc_mem, refc_current_sub_index, refc_nav, caption,
                                                                                    get_observation_func, subtask_judge_func, advice=advice)
            

            new_navigation_lines, new_is_actions, caption, summary = get_observations_lines(refc_nav, refc_mem.landmarks,
                                                                                get_observation_func,
                                                                                caption,
                                                                                refc_step_id)
                
            refc_mem.add_global_nav_sequences([{"line": line, "sub_index": refc_current_sub_index} for line in new_navigation_lines])
            refc_mem.add_global_isActions_sequences([{"is_actions": act, "sub_index": refc_current_sub_index} for act in new_is_actions])
            refc_mem.add_summary(summary)  

        if "stop" not in error_action.lower():
            if refc_nav:
                roll_actions = refc_nav.actions[:error_point]
                roll_subtask_judge = refc_mem.get_subtask_judge(error_point-1)
            else:
                roll_actions = org_nav.actions[:error_point]
                roll_subtask_judge = org_mem.get_subtask_judge(error_point-1)

            if len(roll_actions) - 1 != len(roll_subtask_judge):
                print("error_point", error_point)
                print(len(roll_actions) - 1)
                print(len(roll_subtask_judge))
                assert len(roll_actions) - 1 == len(roll_subtask_judge)

 
            refc_nav, refc_mem, refc_current_sub_index, refc_step_id = self.roll_back(roll_actions, refc_mem,
                                                                                      roll_subtask_judge,
                                                                                      1, caption, max_steps, get_observation_func, graph)
            
            refc_nav, refc_mem, refc_current_sub_index, refc_step_id, no_stop, caption = self.run_execute(max_steps, refc_step_id,
                                                                                    refc_mem, refc_current_sub_index, refc_nav, caption,
                                                                                    get_observation_func, subtask_judge_func, advice=advice)
            

            new_navigation_lines, new_is_actions, caption, summary = get_observations_lines(refc_nav, refc_mem.landmarks,
                                                                                get_observation_func,
                                                                                caption,
                                                                                refc_step_id)
                
            refc_mem.add_global_nav_sequences([{"line": line, "sub_index": refc_current_sub_index} for line in new_navigation_lines])
            refc_mem.add_global_isActions_sequences([{"is_actions": act, "sub_index": refc_current_sub_index} for act in new_is_actions])
            refc_mem.add_summary(summary)
            
            not_continue = False

        return refc_nav, refc_mem, refc_current_sub_index, refc_step_id, no_stop, not_continue, True, caption
                    
    def roll_back(self, actions, refc_mem, roll_subtask_judge, current_sub_index, caption, max_steps, get_observation_func, graph):
        if self.is_map2seq:
            action_id = 2
        else:
            action_id = 1
        nav = self.init_nav(refc_mem.instance, graph)
        refc_mem.reset()
        judge_id = 0
        step_id = 0
        while step_id <= max_steps:
            new_navigation_lines, new_is_actions, caption, summary = get_observations_lines(nav, refc_mem.landmarks,
                                                                               get_observation_func,
                                                                               caption,
                                                                               step_id)
            refc_mem.add_global_nav_sequences([{"line": line, "sub_index": current_sub_index} for line in new_navigation_lines])
            refc_mem.add_global_isActions_sequences([{"is_actions": act, "sub_index": current_sub_index} for act in new_is_actions])
            refc_mem.add_summary(summary)
            
            if not self.is_map2seq and step_id == 0:  # No task completion judgment when dataset is touchdown and is the first step
                pass
            else:
                if judge_id < len(roll_subtask_judge):
                    judge_result = roll_subtask_judge[judge_id]
                    judge_id += 1
                else:
                    print("error judge_result, exit")
                    break
                
                if judge_result:
                    if current_sub_index == len(roll_subtask_judge):
                        print("error judge_result")
                        break
                    current_sub_index += 1
                    refc_mem.add_subtask_judge(True)
                else:
                    refc_mem.add_subtask_judge(False)

            step_id = len(nav.actions)

            if action_id < len(actions):
                action = actions[action_id]
                action_id += 1
            else:
                print("rollback complete")
                break
            action, _ = nav.validate_action(action)

            if action == 'stop':
                print(f"error 'stop' action, actions:{actions}")
                break
            print(step_id, ': ', action)
            nav.step(action)

        return nav, refc_mem, current_sub_index, step_id

    def query_next_action(self, prompt, map_path=None):
        predict, usage = self.llm.query(prompt, map_path)
        self.total_tokens += usage
        for _ in range(5):
            if "sorry" in predict.lower():
                print("Response is empty or malformed: {}, retry".format(predict))
                predict, usage = self.llm.query(prompt, map_path)
                self.total_tokens += usage

        next_action = self.extract_result(predict)
        return next_action

    def extract_result(self, predict):
        match = re.search(r'(?i)"?answer"?\s*[:=]\s*"?([a-z_]+)"?', predict, re.IGNORECASE)
        match_reason = re.search(r'"?reason"?\s*[:=]\s*(.*?)(?=(?:\n|$|"answer"))', predict, re.DOTALL)
        if match_reason:
            reason = match_reason.group(1)
            print("reason:", reason)
        else:
            print("can't get reason!")
            print(predict)
        if match:
            result = match.group(1)
            print("result:", result)
            return result
        else:
            print("Failed to extract 'answer' from response: {}".format(predict))
            print("Default forward")
            return 'forward'
        

class Reflector(Agent):
    def __init__(self, llm, prompts_dir, agent_role, is_map2seq):
        self.total_tokens = 0
        self.is_map2seq = is_map2seq
        self.REFLECTION_TIME = 2
        self.prompts_dir = os.path.join(prompts_dir, 'map2seq' if self.is_map2seq else 'touchdown', 'navigation')
        with open(os.path.join(self.prompts_dir, 'reflecting_top3_map.txt')) as f:
            self.reflection_prompt_template = ''.join(f.readlines())
        
        with open(os.path.join(self.prompts_dir, 'reflecting_verify_map.txt')) as f:
            self.prompt_template = ''.join(f.readlines())

        super().__init__(llm, agent_role)

    def get_reflection(self, navigation_task, global_nav_lines, caption, org_nav, landmark_dic, map_path=None):
        global_lines = '\n'.join(global_nav_lines)
        landmark_label = self.get_label(landmark_dic)
        prompt = self.reflection_prompt_template.format(navigation_task, global_lines, landmark_label)

        top_3, top3_usage = self.llm.query(prompt, map_path)
        print("top3:\n", top_3)
        top_3_prompt = self.restructure_prompt(top_3, caption, org_nav)
        self.total_tokens += top3_usage

        print(global_lines)
        verify_prompt = self.prompt_template.format(navigation_task, global_lines, top_3_prompt, landmark_label)

        predict, verify_usage = self.llm.query(verify_prompt, map_path)
        self.total_tokens += verify_usage
        error_point, error_action, advice = self.extract_result(predict)
        return error_point, error_action, advice

    def extract_result(self, predict):
        print(predict)
        match = re.search(r'(?i)"?answer"?\s*[:=]\s*"?([^"]+)"?', str(predict), re.IGNORECASE)
        match_advice = re.search(r'"?advice"?\s*[:=]\s*"?(.*?)"?(?=(?:\n|$|"answer"))', str(predict), re.DOTALL)
        advice = ''
        if match_advice:
            advice = match_advice.group(1)
            print("advice:", advice)
        else:
            print("Can't get advice! default forward")
            advice = "The forward action should be executed here."
        if match:
            result = match.group(1)
            print("answer:", result)
            error_point, error_action = self.split_action(self.clean_answer(result))
            return error_point, error_action, advice
        else:
            print("Failed to extract 'answer' from response: {}".format(predict))
            print("Default stop")
            return None, "stop", advice

    def restructure_prompt(self, input_text, caption, org_nav):
        top_pattern = r'"?top\s*\d+"?\s*[:=]\s*"?([^"]+)"?'  # 捕获top n后的内容（包括灵活处理空格和符号）
        reason_pattern = r'"?reason"?\s*[:=]\s*"?([^"]+)"?'  # 捕获reason后的内容
        step_number_pattern = r'step\s+(\d+)'

        top_matches = re.findall(top_pattern, input_text)
        reason_matches = re.findall(reason_pattern, input_text)
        if len(top_matches) != len(reason_matches) or len(top_matches) < 1:
            print("Extracted 'top' entries:", top_matches)
            print("Extracted 'reason' entries:", reason_matches)
            print("Mismatch between extracted 'top' and 'reason' entries, default stop")
            top_matches = [f"step {len(org_nav.states)-1}. stop"]
            reason_matches = ["Failure to stop in the correct position. Should try to take a few more steps forward."]
        
        output = []
        for i in range(len(top_matches)):
            top_text = top_matches[i]
            step_match = re.search(step_number_pattern, top_text)
            if step_match:
                step_number = int(step_match.group(1))
                print("error_step num:", step_number)
                print("error action:", top_text)
                print(len(org_nav.states))
                
                if not self.is_map2seq and step_number == 1:
                    panoid, heading = org_nav.get_state_from_index(step_number-1)
                    caption_name = panoid + '_' + str(heading) + '_' + 'init'
                else:
                    panoid, heading = org_nav.get_state_from_index(step_number)
                    caption_name = panoid + '_' + str(heading)
                obs_result = next((item for item in caption if caption_name in item), None)
                caption_prompt = concat_caption(obs_result[caption_name]["caption"])
                top_result = f"top {i+1}: {top_text}\nreason: {reason_matches[i]}\n" + caption_prompt 
                output.append(top_result)
            else:
                print("Mismatch between extracted 'top' and 'reason' entries, next")
        return "\n\n".join(output)
    
    def clean_answer(self, answer):
        return re.sub(r'(\d+\.\s\S+?)\s*\.?$', r'\1', answer)

    def split_action(self, content):
        parts = content.split('. ')
        step_number_pattern = r'step\s+(\d+)'

        if len(parts) != 2:
            print(f"Formatting error: {content}. Default stop")
            number = None
            action = "stop"
        else:
            step_match = re.search(step_number_pattern, parts[0])
            if step_match:
                number = int(step_match.group(1))
                action = parts[1]
            else:
                print(f"step_match error: {content}. Default stop")
                number = None
                action = "stop"
        return number, action

    def get_label(self, landmark_dic):
        label = '\n'.join([f"{key}: {value}" for key, value in landmark_dic.items()])
        return label

class Observer(Agent):
    def __init__(self, llm, prompts_dir, image_dir, agent_role, is_map2seq):
        
        self.image_dir = image_dir
        self.is_map2seq = is_map2seq
        self.prompts_dir = os.path.join(prompts_dir, 'observation')
        with open(os.path.join(self.prompts_dir, 'caption_generate.txt')) as f:
            self.caption_template = ''.join(f.readlines())

        with open(os.path.join(self.prompts_dir, 'landmark_match.txt')) as f:
            self.landmark_template = ''.join(f.readlines())

        with open(os.path.join(self.prompts_dir, 'landmark_match_8.txt')) as f:
            self.landmark_template_8 = ''.join(f.readlines())

        with open(os.path.join(self.prompts_dir, 'caption_summarize_8.txt')) as f:
            self.summary_template_8 = ''.join(f.readlines())

        with open(os.path.join(self.prompts_dir, 'caption_summarize_5.txt')) as f:
            self.summary_template = ''.join(f.readlines())

        with open(os.path.join(prompts_dir, 'observation', 'traffic_flow_judge.txt')) as f:
            self.traffic_flow_template = ''.join(f.readlines())

        self.angles = [-105, -55, 0, 55, 105]
        self.angles_8 = [-135, -90, -45, 0, 45, 90, 135, 180]

        self.fov = 60
        self.height = 700
        self.width = 900

        super().__init__(llm, agent_role)

    def landmark_match(self, recongnized_landmarks, cap_left, cap_left_front, cap_front, cap_right_front, cap_right):
        landmark_prompt = self.landmark_template.format(recongnized_landmarks, cap_left, cap_left_front, cap_front, cap_right_front, cap_right)
        obs_landmark = self.llm.predict(landmark_prompt)

        return obs_landmark.strip()

    def landmark_match_8(self, recongnized_landmarks, cap_left_rear, cap_left, cap_left_front, cap_front, cap_right_front, cap_right, cap_right_rear, cap_rear):
        landmark_prompt = self.landmark_template_8.format(recongnized_landmarks, cap_left_rear, cap_left, cap_left_front, cap_front, cap_right_front, cap_right, cap_right_rear, cap_rear)
        obs_landmark = self.llm.predict(landmark_prompt)

        return obs_landmark.strip()


    def caption_generate(self, pano_name, pano_yaw, heading):
        pano_path = os.path.join(self.image_dir, pano_name + ".jpg")
        images = get_pano_slices(self.angles, self.fov, self.height, self.width, pano_path, pano_yaw, heading)
        ang_obs = list()
        for i in range(5):
            obs = self.llm.predict(self.caption_template, image=images[i])
            ang_obs.append(obs.strip())
        return ang_obs[0], ang_obs[1], ang_obs[2], ang_obs[3], ang_obs[4]

    def caption_generate_8(self, pano_name, pano_yaw, heading):
        pano_path = os.path.join(self.image_dir, pano_name + ".jpg")
        images = get_pano_slices(self.angles_8, self.fov, self.height, self.width, pano_path, pano_yaw, heading)
        ang_obs = list()
        for i in range(8):
            obs = self.llm.predict(self.caption_template, image=images[i])
            ang_obs.append(obs.strip())
        return ang_obs[0], ang_obs[1], ang_obs[2], ang_obs[3], ang_obs[4], ang_obs[5], ang_obs[6], ang_obs[7]

    def summarize_caption_8(self, cap_left_rear, cap_left, cap_left_front, cap_front, cap_right_front, cap_right, cap_right_rear, cap_rear):
        summarize_prompt = self.summary_template_8.format(cap_left_rear, cap_left, cap_left_front, cap_front, cap_right_front, cap_right, cap_right_rear, cap_rear)
        summary = self.llm.predict(summarize_prompt)
        return summary

    def summarize_caption(self, cap_left, cap_left_front, cap_front, cap_right_front, cap_right):
        summarize_prompt = self.summary_template.format(cap_left, cap_left_front, cap_front, cap_right_front, cap_right)
        summary = self.llm.predict(summarize_prompt)
        return summary

    def traffic_flow_judge(self, pano_name, pano_yaw, heading):
        pano_path = os.path.join(self.image_dir, pano_name + ".jpg")
        images = get_pano_slices([0, 180], 90, 750, 800, pano_path, pano_yaw, heading)
        traffic_flow = self.llm.traffic_flow_judge(self.traffic_flow_template, images)
        
        return traffic_flow

