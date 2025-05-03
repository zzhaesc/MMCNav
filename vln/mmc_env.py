import numpy as np
import math
import sys

from vln.graph_loader import GraphLoader
from vln.prompt_builder import get_observations_lines, concat_caption

from vln.base_memory import Memory, mapMaker

sys.path.append('../')

from PIL import Image

import requests
from PIL import Image
from io import BytesIO


class mmcEnv:
    def __init__(self, planner, executor, reflector, Observer, is_map2seq, graph_dir, image_dir, map_dir):
        self.graph = GraphLoader(graph_dir).construct_graph()

        self.Observer = Observer
        self.planner = planner
        self.executor = executor
        self.reflector = reflector

        self.is_map2seq = is_map2seq

        self.image_dir = image_dir
        self.map_dir = map_dir

        self.seen_panoids = {n.panoid for n in self.graph.nodes.values() if n.partition == 'seen'}

    def run(self, max_steps, instance, caption):
        global_nav_sequences = list()
        global_isActions_sequences = list()
        subtask_judge = list()

        nav = self.executor.init_nav(instance, self.graph)
        task_list = self.planner.get_navigation_plan(instance, self.is_map2seq)
        
        mapmaker = mapMaker(self.graph)
        mem = Memory(instance, task_list, 
                     global_nav_sequences, global_isActions_sequences, subtask_judge,
                     mapmaker, self.map_dir)
        
        no_stop = True
        step_id = 0
        current_sub_index = 1

        nav, mem, current_sub_index, step_id, no_stop, caption = self.executor.run_execute(max_steps, step_id,
                                                                                    mem, current_sub_index, nav, caption,
                                                                                    self.get_observations, self.planner.subtask_judge)

        if no_stop:
            nav.step('stop')
            mem.add_subtask_judge(True)

        new_navigation_lines, new_is_actions, caption, summary = get_observations_lines(nav, mem.landmarks,
                                                                                self.get_observations,
                                                                                caption,
                                                                                step_id)

        mem.add_global_nav_sequences([{"line": line, "sub_index": current_sub_index} for line in new_navigation_lines])
        mem.add_global_isActions_sequences([{"is_actions": act, "sub_index": current_sub_index} for act in new_is_actions])
        mem.add_summary(summary)

        if_update = False

        refc_nav = self.executor.copy_nav(nav.actions, instance, self.graph)

        refc_mem = Memory(instance, task_list, 
                     mem.global_nav_sequences, mem.global_isActions_sequences, mem.subtask_judge,
                     mapmaker, self.map_dir, mem.global_summary)
        
        refc_current_sub_index = current_sub_index

        not_continue = False
        for _ in range(self.reflector.REFLECTION_TIME):
            if not_continue:
                break
            else:
                _ = refc_mem.get_map(refc_nav)
                refc_nav, refc_mem, refc_current_sub_index, step_id, no_stop, not_continue, if_update, caption = self.executor.retry(refc_nav,
                                                                                                                                    mem,
                                                                                                                                    refc_mem,
                                                                                                                                    refc_current_sub_index,
                                                                                                                                    caption,
                                                                                                                                    max_steps,
                                                                                                                                    self.reflector.get_reflection,
                                                                                                                                    self.planner.subtask_judge,
                                                                                                                                    self.get_observations,
                                                                                                                                    self.graph)
                if no_stop:
                    refc_nav.step('stop')
                    refc_mem.add_subtask_judge(True)

                new_navigation_lines, new_is_actions, caption, summary = get_observations_lines(refc_nav, refc_mem.landmarks,
                                                                                self.get_observations,
                                                                                caption,
                                                                                step_id)
                
                refc_mem.add_global_nav_sequences([{"line": line, "sub_index": refc_current_sub_index} for line in new_navigation_lines])
                refc_mem.add_global_isActions_sequences([{"is_actions": act, "sub_index": refc_current_sub_index} for act in new_is_actions])
                refc_mem.add_summary(summary)
                
                if if_update:
                    del nav
                    del mem
                    nav = self.executor.copy_nav(refc_nav.actions, instance, self.graph)
                    mem = Memory(instance, task_list, 
                                refc_mem.global_nav_sequences, refc_mem.global_isActions_sequences, refc_mem.subtask_judge,
                                mapmaker, self.map_dir, refc_mem.global_summary)
                    print(len(nav.actions))
                    print(nav.actions)
                    print(len(mem.subtask_judge))
                    print(mem.subtask_judge)
                else:
                    del refc_nav
                    del refc_mem
                    refc_nav = self.executor.copy_nav(nav.actions, instance, self.graph)
                    refc_mem = Memory(instance, task_list, 
                                mem.global_nav_sequences, mem.global_isActions_sequences, mem.subtask_judge,
                                mapmaker, self.map_dir, mem.global_summary)
                    print(len(refc_nav.actions))
                    print(refc_nav.actions)
                    print(len(refc_mem.subtask_judge))
                    print(refc_mem.subtask_judge)
                if not_continue:
                    print("out of max steps or destination")
                    break

        return nav, mem.get_global_nav_lines(), mem.get_global_is_actions(), mem.get_subtask_judge(), mem.taskPools, caption

    def get_observations(self, states, step_id, landmarks, caption_sets):
        observations = dict()
        observations['is_map2seq'] = self.is_map2seq
        state = states[step_id]
        prev_state = None, None
        caption = dict()
        if step_id > 0:
            prev_state = states[step_id-1]

        panoid, heading = state
        prev_panoid, _ = prev_state
        num_neighbors = self.graph.get_num_neighbors(panoid)
        if num_neighbors > 2 and panoid != prev_panoid:
            observations['intersection'] = num_neighbors
        
        traffic_flow = None

        if not self.is_map2seq and step_id == 0:
            caption_name = panoid + '_' + str(heading) + '_' + 'init'
            obs_result = next((item for item in caption_sets if caption_name in item), None)
            recongnized_landmarks = '\n'.join(landmarks)

            if obs_result:
                cap_left_rear = obs_result[caption_name]["caption"][0]
                cap_left = obs_result[caption_name]["caption"][1]
                cap_left_front = obs_result[caption_name]["caption"][2]
                cap_front = obs_result[caption_name]["caption"][3]
                cap_right_front = obs_result[caption_name]["caption"][4]
                cap_right = obs_result[caption_name]["caption"][5]
                cap_right_rear = obs_result[caption_name]["caption"][6]
                cap_rear = obs_result[caption_name]["caption"][7]

                summary = obs_result[caption_name]["summary"]
                traffic_flow = obs_result[caption_name]["traffic_flow"]
                
            else:
                caption[caption_name] = dict()
                pano_yaw = self.graph.nodes[panoid].pano_yaw_angle

                cap_left_rear, cap_left, cap_left_front, cap_front, cap_right_front, cap_right, cap_right_rear, cap_rear = self.Observer.caption_generate_8(panoid, pano_yaw, round(heading))
                summary = self.Observer.summarize_caption(cap_left_rear, cap_left, cap_left_front, cap_front, cap_right_front, cap_right, cap_right_rear, cap_rear)
                traffic_flow = self.Observer.traffic_flow_judge(panoid, pano_yaw, round(heading))

                caption[caption_name]['traffic_flow'] = traffic_flow.strip()
                caption[caption_name]['caption'] = [cap_left_rear, cap_left, cap_left_front, cap_front, cap_right_front, cap_right, cap_right_rear, cap_rear]
                caption[caption_name]['summary'] = summary.strip()
            
            obs_landmarks = self.Observer.landmark_match_8(recongnized_landmarks, 
                                                         cap_left_rear, cap_left,
                                                         cap_left_front, cap_front,
                                                         cap_right_front, cap_right, 
                                                         cap_right_rear, cap_rear)

        else:
            caption_name = panoid + '_' + str(heading)

            obs_result = next((item for item in caption_sets if caption_name in item), None)
            recongnized_landmarks = '\n'.join(landmarks)
            if obs_result:
                cap_left = obs_result[caption_name]["caption"][0]
                cap_left_front = obs_result[caption_name]["caption"][1]
                cap_front = obs_result[caption_name]["caption"][2]
                cap_right_front = obs_result[caption_name]["caption"][3]
                cap_right = obs_result[caption_name]["caption"][4]
                
                summary = obs_result[caption_name]["summary"]
                
            else:
                caption[caption_name] = dict()
                pano_yaw = self.graph.nodes[panoid].pano_yaw_angle

                cap_left, cap_left_front, cap_front, cap_right_front, cap_right = self.Observer.caption_generate(panoid, pano_yaw, round(heading))
                summary = self.Observer.summarize_caption(cap_left, cap_left_front, cap_front, cap_right_front, cap_right)

                caption[caption_name]['caption'] = [cap_left, cap_left_front, cap_front, cap_right_front, cap_right]
                caption[caption_name]['summary'] = summary.strip()

            obs_landmarks = self.Observer.landmark_match(recongnized_landmarks, 
                                                        cap_left, cap_left_front, cap_front,
                                                        cap_right_front, cap_right)
            
        if traffic_flow is not None:
            observations['traffic_flow'] = traffic_flow
        observations['landmarks'] = obs_landmarks
        observations['summary'] = summary.strip()

        return observations, caption

    def get_observations_noimage(self, states, step_id):
        observations = dict()
        observations['is_map2seq'] = self.is_map2seq
        state = states[step_id]
        prev_state = None, None
        if step_id > 0:
            prev_state = states[step_id-1]

        panoid, heading = state
        prev_panoid, _ = prev_state

        # intersection
        num_neighbors = self.graph.get_num_neighbors(panoid)
        if num_neighbors > 2 and panoid != prev_panoid:
            observations['intersection'] = num_neighbors


        return observations
class navState:
    def __init__(self, graph):
        self.action_list = ["forward", "left", "right", "stop", "turn_around"]
        self.action_mapping = {'turnaround': 'turn_around', 'turn around': 'turn_around', 'turn_left': 'left',
                      'turn left': 'left', 'turn_right': 'right', 'turn right': 'right'}
        
        self.graph = graph
        self.states = list()
        self.actions = list()
        self.pano_path = list()

    def init_state(self, panoid, heading):
        self._set_state(panoid, heading)
        self.actions.append('init')

    def _set_state(self, panoid, heading):
        heading = round(heading)
        state = (panoid, heading)

        prev_pano, _ = self.get_state()
        if panoid != prev_pano:
            self.pano_path.append(panoid)

        self.states.append(state)

    def get_state(self):
        if len(self.states) == 0:
            return None, None
        return self.states[-1]
    
    def get_prev_state(self):
        if len(self.states) < 2:
            return None, None
        return self.states[-2]
    
    def get_state_from_index(self, step_number):
        if step_number >= len(self.states):
            print("states len:", len(self.states))
            print("step_number:", step_number)
            assert step_number < len(self.states)
        return self.states[step_number]
    
    def validate_action(self, action):
        curr_panoid, _ = self.get_state()
        num_neighbors = self.graph.get_num_neighbors(curr_panoid)

        action = self.action_mapping.get(action, action)
        if num_neighbors <= 1:
            if action not in ['stop', 'turn_around']:
                print("Stop at the edge of the map")
                return 'stop', True
        if num_neighbors == 2:  # can only stop, turn_around or forward on regular street segment
            if action in ['left', 'right']:
                print("rectification forward")
                return 'forward', False

        if action not in self.action_list:
            print('action that caused error:', action)

            action = 'forward'
        return action, False
    
    def step(self, action):
        if action == 'init':
            return
        assert action in self.action_list

        next_panoid, next_heading = self.get_next_state(action)
        self.actions.append(action)
        self._set_state(next_panoid, next_heading)

    def get_heading_form_states(self, target_panoid):
        for panoid, heading in self.states:
            if panoid == target_panoid:
                return heading
        return None
    
    def get_next_state(self, action):
        curr_pano, curr_heading = self.get_state()
        prev_pano, _ = self.get_prev_state()

        neighbors = self.graph.nodes[curr_pano].neighbors
        num_neighbors = len(neighbors)
        out_headings = list(neighbors.keys())

        if action == "stop":
            return curr_pano, curr_heading

        if action == 'forward' and curr_heading in neighbors:
            return neighbors[curr_heading].panoid, curr_heading

        if action == 'turn_around':
            out_heading = (curr_heading - 180) % 360
            next_heading = get_closest_heading(out_heading, out_headings)
            return curr_pano, next_heading

        if num_neighbors <= 1 and prev_pano not in [None, curr_pano]:
            # don't move if end of graph reached; but only if agent is not at starting point.
            return curr_pano, curr_heading

        # regular street segment
        if num_neighbors == 2:
            next_heading = get_closest_heading(curr_heading, out_headings)

        # handle intersection
        if num_neighbors > 2:
            next_heading = self._get_next_heading_intersection(action)

        if action == 'forward':
            next_pano = neighbors[next_heading].panoid
        else:  # "left" or "right" rotates the agent but does not move panos
            next_pano = curr_pano
        return next_pano, next_heading

    def forward_exploration(self, action, curr_pano, curr_heading, prev_pano, prev_heading):
        neighbors = self.graph.nodes[curr_pano].neighbors
        num_neighbors = len(neighbors)
        out_headings = list(neighbors.keys())

        next_heading = curr_heading
        if action == "stop":
            return curr_pano, curr_heading, curr_pano, curr_heading

        if action == 'forward' and curr_heading in neighbors:
            return neighbors[curr_heading].panoid, curr_heading, curr_pano, curr_heading

        if action == 'turn_around':
            out_heading = (curr_heading - 180) % 360
            next_heading = get_closest_heading(out_heading, out_headings)
            return curr_pano, next_heading, curr_pano, curr_heading

        if num_neighbors <= 1 and prev_pano not in [None, curr_pano]:
            # don't move if end of graph reached; but only if agent is not at starting point.
            return curr_pano, curr_heading, curr_pano, curr_heading

        # regular street segment
        if num_neighbors == 2:
            next_heading = get_closest_heading(curr_heading, out_headings)

        # handle intersection
        if num_neighbors > 2:
            next_heading = self.forward_intersection_exploration(action, curr_pano, curr_heading, prev_pano, prev_heading)

        if action == 'forward' and num_neighbors > 1:
            # print(next_heading)
            next_pano = neighbors[next_heading].panoid
        else:  # "left" or "right" rotates the agent but does not move panos
            next_pano = curr_pano
        return next_pano, next_heading, curr_pano, curr_heading
    
    def forward_intersection_exploration(self, action, curr_pano, curr_heading, prev_pano, prev_heading):
        curr_node = self.graph.nodes[curr_pano]
        neighbors = curr_node.neighbors

        # heading of all outgoing edges
        out_headings = list(neighbors.keys())

        forward_heading = curr_heading
        if curr_pano != prev_pano and prev_pano is not None:
            out_headings.remove(curr_node.get_neighbor_heading(prev_pano))   # Remove the edge containing the previous state

            # select forward_heading relative to other outgoing edges
            out_headings_sorted = list(sorted(out_headings, key=lambda h: get_relative_angle(curr_heading, h)))

            # forward_heading is the middle direction of all outgoing edges
            n = len(out_headings_sorted)
            if len(out_headings_sorted) % 2 == 0:
                forward_heading_1 = out_headings_sorted[n // 2 - 1]
                forward_heading_2 = out_headings_sorted[n // 2]
                forward_heading = get_closest_heading(curr_heading, [forward_heading_1, forward_heading_2], prev_heading)
            else:
                forward_heading = out_headings_sorted[n // 2]

            if len(neighbors) == 3:
                if action == 'left':
                    return out_headings_sorted[0]
                if action == 'right':
                    return out_headings_sorted[-1]

        if action == 'forward':
            return forward_heading

        candidate_headings = set(out_headings) - {forward_heading}
        if action == 'left':
            return min(candidate_headings, key=lambda h: (forward_heading - h) % 360)
        if action == 'right':
            return min(candidate_headings, key=lambda h: (h - forward_heading) % 360)
        
    def _get_next_heading_intersection(self, action):
        curr_pano, curr_heading = self.get_state()
        prev_pano, prev_heading = self.get_prev_state()
        curr_node = self.graph.nodes[curr_pano]
        neighbors = curr_node.neighbors

        # heading of all outgoing edges
        out_headings = list(neighbors.keys())

        forward_heading = curr_heading
        if curr_pano != prev_pano and prev_pano is not None:
            out_headings.remove(curr_node.get_neighbor_heading(prev_pano))

            # select forward_heading relative to other outgoing edges
            out_headings_sorted = list(sorted(out_headings, key=lambda h: get_relative_angle(curr_heading, h)))

            # forward_heading is the middle direction of all outgoing edges
            n = len(out_headings_sorted)
            if len(out_headings_sorted) % 2 == 0:
                forward_heading_1 = out_headings_sorted[n // 2 - 1]
                forward_heading_2 = out_headings_sorted[n // 2]
                forward_heading = get_closest_heading(curr_heading, [forward_heading_1, forward_heading_2], prev_heading)
            else:
                forward_heading = out_headings_sorted[n // 2]

            if len(neighbors) == 3:
                if action == 'left':
                    return out_headings_sorted[0]
                if action == 'right':
                    return out_headings_sorted[-1]

        if action == 'forward':
            return forward_heading

        candidate_headings = set(out_headings) - {forward_heading}
        if action == 'left':
            return min(candidate_headings, key=lambda h: (forward_heading - h) % 360)
        if action == 'right':
            return min(candidate_headings, key=lambda h: (h - forward_heading) % 360)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out
    
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image.resize((1500,750), Image.LANCZOS)

def get_nav_from_actions(actions, instance, graph):
    nav = navState(graph)

    start_heading = instance['start_heading']
    start_panoid = instance['route_panoids'][0]
    nav.init_state(panoid=start_panoid,
                   heading=start_heading)

    for i, action in enumerate(actions):
        nav.step(action)

    assert nav.actions == actions
    return nav

def get_gold_nav(instance, graph):
    nav = navState(graph)

    start_heading = instance['start_heading']
    start_panoid = instance['route_panoids'][0]
    nav.init_state(panoid=start_panoid,
                   heading=start_heading)

    gt_action = None
    while gt_action != 'stop':
        gt_action = get_gt_action(nav, gt_path=instance['route_panoids'])
        nav.step(gt_action)

    test_nav = navState(graph)
    test_nav.init_state(panoid=start_panoid,
                        heading=start_heading)

    for action in nav.actions:
        test_nav.step(action)

    assert nav.pano_path == instance['route_panoids']
    assert nav.get_state() == test_nav.get_state()
    assert nav.get_state()[0] == instance['route_panoids'][-1]
    return nav

def get_gt_action(nav, gt_path):
    target_panoid = gt_path[-1]
    curr_panoid, _ = nav.get_state()
    curr_node = nav.graph.nodes[curr_panoid]

    if curr_panoid in gt_path:
        num_occurrences = gt_path.count(curr_panoid)
        if num_occurrences == 1:
            pano_index = gt_path.index(curr_panoid)
        else:  # if novel gold path visits panoid twice then select the correct one based on the current trajectory
            num_occurrences_nav = nav.pano_path.count(curr_panoid)
            nth_occurrence = min(num_occurrences, num_occurrences_nav)-1
            pano_index = [i for i, p in enumerate(gt_path) if p == curr_panoid][nth_occurrence]

        if pano_index == len(gt_path)-1:
            assert gt_path[pano_index] == target_panoid
            return 'stop'

        gt_next_panoid = gt_path[pano_index + 1]
        gt_next_heading = curr_node.get_neighbor_heading(gt_next_panoid)
    else:
        shortest_path = nav.graph.get_shortest_path(curr_panoid, target_panoid)
        if len(shortest_path) <= 1:
            return 'stop'
        gt_next_panoid = shortest_path[1]
        gt_next_heading = curr_node.get_neighbor_heading(gt_next_panoid)

    next_panoid, next_heading = nav.get_next_state('forward')
    if gt_next_panoid == next_panoid:
        # at 3-way intersection, "forward" AND "left"/"right" can be correct. Only chose forward as gold action
        # if it doesn't imply a rotation of over 45 degrees.
        if len(curr_node.neighbors) != 3 or abs(get_relative_angle(next_heading, gt_next_heading)) < 45:
            return 'forward'

    next_panoid, next_heading = nav.get_next_state('turn_around')
    if gt_next_heading == next_heading:
        return 'turn_around'

    next_panoid, next_heading_left = nav.get_next_state('left')
    if gt_next_heading == next_heading_left:
        return 'left'

    next_panoid, next_heading_right = nav.get_next_state('right')
    if gt_next_heading == next_heading_right:
        return 'right'

    # if multiple rotations are needed, choose direction which brings the agent closer to the correct next heading
    next_heading = get_closest_heading(gt_next_heading, [next_heading_left, next_heading_right])
    if next_heading == next_heading_left:
        return 'left'
    if next_heading == next_heading_right:
        return 'right'

    raise ValueError('gt_action_found not found')

def get_closest_heading(heading, headings, prev_heading=None):
    if prev_heading == None:
        closest = min(headings, key=lambda h: 180 - abs(abs(heading - h) - 180))
    else:
        avg_heading = average_angles([heading, prev_heading], [0.4, 0.6])
        closest = min(headings, key=lambda h: 180 - abs(abs(avg_heading - h) - 180))
    return closest

def average_angles(angles, weights):
    sin_sum = 0
    cos_sum = 0

    for angle, weight in zip(angles, weights):
        angle_radians = math.radians(angle)
        sin_sum += math.sin(angle_radians) * weight
        cos_sum += math.cos(angle_radians) * weight

    average_angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    return average_angle

def angle_to_vector(angle):
    radians = math.radians(angle)
    return (math.cos(radians), math.sin(radians))

def vector_to_angle(vector):
    return math.degrees(math.atan2(vector[1], vector[0])) % 360

def get_relative_angle(curr_heading, heading):
    angle = heading - curr_heading
    if angle > 180:
        angle = angle - 360
    if angle <= -180:
        angle = angle + 360
    return angle
