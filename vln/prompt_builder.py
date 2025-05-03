import re

action_space = 'Action Space:\nforward (go straight), left (rotate left), right (rotate right), stop (end navigation)\n\n'
prompt_template = 'Navigation Instructions:\n"{}"\nAction Sequence:\n'


def build_prompt(instructions):
    prompt = action_space
    prompt += prompt_template.format(instructions, action_space)
    return prompt

  
def get_navigation_lines_baseline(nav, env, landmarks, traffic_flow, is_map2seq=True, step_id=0):
    actions = nav.actions
    states = nav.states

    assert len(actions) == len(states)

    lines = list()
    is_action = list()
    while step_id < len(actions):
        action = actions[step_id]

        # print step number and action
        line = f'{step_id}. {action}'
        if action != 'init':
            lines.append(line)
            is_action.append(True)

        observations = env.get_observations(states, step_id, landmarks, is_map2seq, traffic_flow)
        observations_str = get_observations_str_baseline(observations)

        if observations_str:
            line = observations_str
            lines.append(line)
            is_action.append(False)

        step_id += 1

    if actions[-1] != 'stop':
        line = f'{len(actions)}. '
        lines.append(line)
        is_action.append(False)

    assert len(lines) == len(is_action)
    # print("lines:\n", lines)
    return lines, is_action


def get_observations_str_baseline(observations):
    observations_strs = list()

    if 'traffic_flow' in observations:
        traffic_flow = observations["traffic_flow"]
        observations_strs.append(f'You are aligned {traffic_flow} the flow of the traffic.')

    if 'intersection' in observations:
        num_ways = observations['intersection']
        observations_strs.append(f'There is a {num_ways}-way intersection.')

    if 'landmarks' in observations:
        directions = ['on your left', 'slightly left', 'ahead', 'slightly right', 'on your right']
        for direction, landmarks in zip(directions, observations['landmarks']):
            if len(landmarks) > 0:
                landmarks = ' and '.join(landmarks)
                landmarks = landmarks[0].upper() + landmarks[1:]
                observations_strs.append(f'{landmarks} {direction}.')

    return ' '.join(observations_strs)


def get_observations_lines_noimage(nav, get_observation_func, step_id=0):
    actions = nav.actions
    states = nav.states

    assert len(actions) == len(states)

    lines = list()
    is_action = list()
    while step_id < len(actions):

        action = actions[step_id]
        line = f'{step_id}. {action}'
        if action != 'init':
            lines.append(line)
            is_action.append(True)

        observations = get_observation_func(states, step_id)
        observations_str = get_observations_noimage(observations)

        if observations_str:
            line = observations_str
            lines.append(line)
            is_action.append(False)

        step_id += 1

    assert len(lines) == len(is_action)

    return lines, is_action


def get_observations_noimage(observations):
    observations_strs = list()

    if 'traffic_flow' in observations:
        traffic_flow = observations["traffic_flow"]
        observations_strs.append(f'You are aligned {traffic_flow} the flow of the traffic. ')

    if 'intersection' in observations:
        num_ways = observations['intersection']
        observations_strs.append(f'Intersection Information: There is a {num_ways}-way intersection. ')

    if 'landmarks' in observations:
        directions = ['on your left', 'slightly left', 'ahead', 'slightly right', 'on your right']
        for direction, landmark in zip(directions, observations['landmarks']):
            if len(landmark) > 0:
                landmark = ' and '.join(landmark)
                landmark = landmark[0].upper() + landmark[1:]
                observations_strs.append(f'Landmark Information: {landmark} {direction}. ')

    return ''.join(observations_strs)


def get_observations_lines(nav, landmarks, get_observation_func, caption, step_id=0):
    actions = nav.actions
    states = nav.states

    assert len(actions) == len(states)

    lines = list()
    is_action = list()
    summary = list()
    while step_id < len(actions):
        action = actions[step_id]

        # print step number and action
        line = f'step {step_id}. {action}'
        if action != 'init':
            lines.append(line)
            is_action.append(True)

        observations, cap = get_observation_func(states, step_id, landmarks, caption)

        if len(cap) > 0:
            caption.append(cap)

        summary.append(observations['summary'])

        observations_str = get_observations_str(landmarks, observations)

        if observations_str:
            line = observations_str
            lines.append(line)
            is_action.append(False)

        step_id += 1

    assert len(lines) == len(is_action)
    return lines, is_action, caption, summary

def get_observations_str(landmarks, observations):
    observations_strs = list()
    if 'traffic_flow' in observations:
        traffic_flow = observations["traffic_flow"]
        observations_strs.append(traffic_flow)

    if 'intersection' in observations:
        num_ways = observations['intersection']
        observations_strs.append(f'Intersection Information: There is a {num_ways}-way intersection. ')
    
    if 'landmarks' in observations:
        for line in observations['landmarks'].strip().split('\n'):
            if ': ' in line:
                landmark, direction = line.split(': ', 1)
                if landmark in landmarks:
                    if direction.lower() == 'left rear':
                        observations_strs.append(f'Landmark Information: {landmark} on your left rear. ')
                    if direction.lower() == 'left':
                        observations_strs.append(f'Landmark Information: {landmark} on your left. ')
                    if direction.lower() == 'left front':
                        observations_strs.append(f'Landmark Information: {landmark} slightly left. ')
                    if direction.lower() == 'front':
                        observations_strs.append(f'Landmark Information: {landmark} ahead. ')
                    if direction.lower() == 'right front':
                        observations_strs.append(f'Landmark Information: {landmark} slightly right. ')
                    if direction.lower() == 'right':
                        observations_strs.append(f'Landmark Information: {landmark} on your right. ')
                    if direction.lower() == 'right rear':
                        observations_strs.append(f'Landmark Information: {landmark} on your right rear. ')
                    if direction.lower() == 'rear':
                        observations_strs.append(f'Landmark Information: {landmark} behind. ')
    return ''.join(observations_strs)
  

def concat_caption(caption):
    caption_prompt = ''
    if len(caption) == 5:
        caption_prompt = "Observations in the left:\n" + caption[0] + "\n" + \
        "Observations in the left front:\n" + caption[1] + "\n" + \
        "Observations directly front:\n" + caption[2] + "\n" + \
        "Observations in the right front:\n" + caption[3] + "\n" + \
        "Observations in the right:\n" + caption[4]
    else:
        caption_prompt = "Observations in the left rear direction:\n" + caption[0] + "\n" + \
        "Observations in the left:\n" + caption[1] + "\n" + \
        "Observations in the left front:\n" + caption[2] + "\n" + \
        "Observations directly front:\n" + caption[3] + "\n" + \
        "Observations in the right front:\n" + caption[4] + "\n" + \
        "Observations in the right:\n" + caption[5] + "\n" + \
        "Observations in the right rear direction:\n" + caption[6] + "\n" + \
        "Observations directly rear:\n" + caption[7]
    return caption_prompt

def join_summary(obs_summary):
    summary = ''
    pattern_left = r'(?:Left|left)[^:]*:\s*(.*?)(?:\n|$)'
    pattern_ahead = r'(?:Ahead|ahead)[^:]*:\s*(.*?)(?:\n|$)'
    pattern_right = r'(?:Right|right)[^:]*:\s*(.*?)(?:\n|$)'
    
    left_content = re.search(pattern_left, obs_summary)
    ahead_content = re.search(pattern_ahead, obs_summary)
    right_content = re.search(pattern_right, obs_summary)

    left_sum = left_content.group(1).strip() if left_content else None
    ahead_sum = ahead_content.group(1).strip() if ahead_content else None
    right_sum = right_content.group(1).strip() if right_content else None
    
    if left_sum:
        summary += 'On the left, ' + left_sum
    else:
        summary += 'There is no landmark on the left.'

    if ahead_sum:
        summary += ' Ahead, ' + ahead_sum
    else:
        summary += ' There is no landmark ahead.'

    if right_sum:
        summary += ' On the right, ' + left_sum
    else:
        summary += ' There is no landmark on the right.'

    return summary