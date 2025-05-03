import os
import json
import numpy as np
import time
import re
import sys
import cv2
from PIL import Image

from bokeh.plotting import figure
from bokeh.io.export import export_png
from bokeh.models import ColumnDataSource, LabelSet, Arrow, VeeHead

from tqdm import tqdm
import requests
import asyncio



class Memory:
    def __init__(self, instance, task_list, global_nav_sequences, global_isActions_sequences, subtask_judge, mapMaker, map_dir, global_summary=None):
        self.instance = instance
        self.landmarks = instance['landmarks']
        self.landmarks_dic = { i + 1: self.landmarks[i] for i in range(len(self.landmarks))}

        self.instance_id = instance['id']
        self.dataset_name = instance['dataset_name']
        self.traffic_flow = instance.get('traffic_flow')
        self.navigation_text = instance["navigation_text"]
        self.final_state = instance['final_state']

        self.task_list = task_list
        self.taskPools = self.get_task_dict(task_list)
        self.global_nav_sequences = global_nav_sequences
        self.global_isActions_sequences = global_isActions_sequences
        self.subtask_judge = subtask_judge

        if global_summary:
            self.global_summary = global_summary
        else:
            self.global_summary = list()

        self.reflection_map_dir = map_dir

        self.mapMaker = mapMaker


    def get_map(self, nav):
        filename = f"{self.instance_id}_reflection.png"
        return self.mapMaker.draw(self.get_global_nav_lines(),
                                      self.landmarks_dic, nav, self.taskPools, self.subtask_judge,
                                      self.reflection_map_dir, filename
                                      )

    def get_map_path(self):
        filename = f"{self.instance_id}_reflection.png"
        return os.path.join(self.reflection_map_dir, filename)


    def get_subtask_lines(self, subtask_index):

        filtered_lines = [entry for entry in self.global_nav_sequences if entry["sub_index"] == subtask_index]
        renumbered_lines = list()

        i = 1
        for entry in filtered_lines:

            line_content = entry["line"]
            first_space_index = line_content.find('. ')
            if first_space_index != -1 and first_space_index < 8:
                new_line_content = f"step {i}.{line_content[first_space_index + 1:]}"
                i = i + 1
            else:
                new_line_content = f"{line_content}"

            renumbered_entry = new_line_content
            renumbered_lines.append(renumbered_entry)

        return renumbered_lines
    
    def get_global_nav_lines(self):
        return [entry["line"] for entry in self.global_nav_sequences]
    
    def get_global_is_actions(self):
        return [entry["is_actions"] for entry in self.global_isActions_sequences]
    
    def get_subtask_judge(self, index=None):
        if index is not None:
            return self.subtask_judge[:index]
        else:
            return self.subtask_judge

    def get_task_dict(self, task_list):
        pattern = re.compile(r'^\d+\.\s*(.*)')
        taskPools = []
        
        for line in task_list.split('\n'):
            match = pattern.match(line)
            if match:
                taskPools.append(line)
            
        taskPools = {i+1: command for i, command in enumerate(taskPools)}

        return taskPools

    def add_subtask_judge(self, judge_result, index=None):
        if index is not None:
            self.subtask_judge[index] = judge_result
        else:
            self.subtask_judge.append(judge_result)

    def add_global_nav_sequences(self, new_nav_sequences):
        self.global_nav_sequences += new_nav_sequences
    
    def add_global_isActions_sequences(self, new_isActions_sequences):
        self.global_isActions_sequences += new_isActions_sequences

    def reset(self):
        self.global_nav_sequences = list()
        self.global_isActions_sequences = list()
        self.subtask_judge = list()
        self.global_summary = list()

    def add_summary(self, summary):
        for summ in summary:
            self.global_summary.append(summ)


class mapMaker:
    def __init__(self, graph):
        self.graph = graph
        self.x1, self.y1 = 1, 1
        self.roadWidth = 0.5
        self.elongation = 2.7
        self.nav_len = 0

    def draw(self, global_nav_lines, landmarks_dic, nav, taskPools, subtask_judge, output_dir, filename):
        p = figure(width=500, height=500, title="Navigation Map", toolbar_location=None, match_aspect=True)
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        curr_heading = 0
        set_x = []
        set_y = []
        set_i = []
        set_road = []
        landmarks = []
        set_landmarkx_y = []
        set_landmark = []
        set_landmark_labelx_y = []
        set_road_line = []
        set_intersection_xy = []

        xi = self.x1
        yi = self.y1
        set_x.append(xi)
        set_y.append(yi)
        set_i.append(1)
        i = 1
        gap = 0
        x1_road, y1_road = set_x[-1], set_y[-1]

        panoPath = nav.pano_path
        initPano = panoPath[0]
        self.nav_len = len(global_nav_lines)

        current_sub_index = 1
        judge_id = 0
        for k, item in enumerate(global_nav_lines):
            match = re.match(r'step\s+(\d+)\.\s*(forward|right|left|turn_around|stop)', item)
            if match:
                i = match.group(1)
                action = match.group(2)
                if action == 'forward':
                    xi, yi = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading, 1)
                    set_x.append(xi)
                    set_y.append(yi)
                    set_i.append(int(i)+1)
                    road, road_line1, road_line2 = self.calculate_road(curr_heading, x1_road, y1_road, xi, yi)
                    set_road.append(road)
                    set_road_line.append(road_line1)
                    set_road_line.append(road_line2)
                else:
                    x2_road, y2_road = set_x[-1], set_y[-1]
                    road, road_line1, road_line2 = self.calculate_road(curr_heading, x1_road, y1_road, x2_road, y2_road)
                    set_road.append(road)
                    set_road_line.append(road_line1)
                    set_road_line.append(road_line2)
                    if action == 'stop':
                        break
                    x1_road, y1_road = x2_road, y2_road
                    if action == 'right':
                        curr_heading = (curr_heading + 90 + 360) % 360
                    elif action == 'left':
                        curr_heading = (curr_heading - 90 + 360) % 360
                    else:
                        curr_heading = (curr_heading + 180 + 360) % 360
                    gap += 1
                if judge_id < len(subtask_judge):
                    judge_result = subtask_judge[judge_id]
                    if judge_result:
                        current_sub_index += 1
                        if current_sub_index > len(taskPools):
                            current_sub_index == len(taskPools)
                    judge_id += 1
            else:
                intersection_match = re.search(r'Intersection Information: There is a (\d+)-way intersection.', item)
                if intersection_match:
                    if k == 0:
                        curr_pano = initPano
                    else:
                        curr_pano = None
                    num_ways = intersection_match.group(1)
                    if num_ways == '3':
                        if initPano == curr_pano:
                            curr_node = self.graph.nodes[curr_pano]
                            neighbors = curr_node.neighbors
                            if len(neighbors) != 3:
                                raise Exception(f"node error! no 3-way intersection {curr_pano}")
                            out_headings = list(neighbors.keys())
                            main_heading = nav.get_heading_form_states(curr_pano)
                            mode = self.t_intersection_mode(out_headings, main_heading)
                            road1, road2, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2  = self.draw_3way_intersection(xi, yi, mode, 180)
                        else:
                            curr_pano = panoPath[int(i)-gap]
                            curr_node = self.graph.nodes[curr_pano]
                            neighbors = curr_node.neighbors
                            if len(neighbors) != 3:
                                raise Exception(f"node error! no 3-way intersection {curr_pano}")
                            out_headings = list(neighbors.keys())
                            main_heading = curr_node.get_neighbor_heading(panoPath[int(i)-gap-1])  
                            mode = self.t_intersection_mode(out_headings, main_heading)

                            road1, road2, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2  = self.draw_3way_intersection(xi, yi, mode, curr_heading)

                        set_road.append(road1)
                        set_road.append(road2)
                        set_road_line.append(road_horizontal_line1)
                        set_road_line.append(road_horizontal_line2)
                        set_road_line.append(road_vertical_line1)
                        set_road_line.append(road_vertical_line2)
                        set_intersection_xy.append((xi, yi-0.3))
                    elif num_ways == '4':
                        road1, road2, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2 = self.draw_4way_intersection(xi,yi)
                        set_road.append(road1)
                        set_road.append(road2)
                        set_road_line.append(road_horizontal_line1)
                        set_road_line.append(road_horizontal_line2)
                        set_road_line.append(road_vertical_line1)
                        set_road_line.append(road_vertical_line2)
                        set_intersection_xy.append((xi, yi-0.3))
                    else:
                        road_vertical, road_horizontal, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2, side_road, side_road_line1, side_road_line2 = self.draw_5way_intersection(xi, yi)
                        set_road.append(road_vertical)
                        set_road.append(road_horizontal)
                        set_road.append(side_road)
                        set_road_line.append(road_horizontal_line1)
                        set_road_line.append(road_horizontal_line2)
                        set_road_line.append(road_vertical_line1)
                        set_road_line.append(road_vertical_line2)
                        set_road_line.append(side_road_line1)
                        set_road_line.append(side_road_line2)
                        set_intersection_xy.append((xi, yi-0.3))
                
                landmark_x, landmark_y = set_x[-1], set_y[-1]
                landmark_infos = re.findall(r'Landmark Information: (.+?) (on your left rear|on your left|slightly left|ahead|slightly right|on your right|on your right rear|behind)\.', item)
                for landmark, direction in landmark_infos:
                    draw_landmark = False
                    landmark_index = 0
                    for i in range(len(taskPools)):
                        if landmark.lower() in taskPools[i+1].lower():
                            landmark_index = i+1
                            break
                    if landmark_index <= current_sub_index and landmark.lower() not in [item.lower() for item in landmarks]:
                        draw_landmark = True
                    if draw_landmark:
                            if direction == 'on your left':
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading - 90 + 360) % 360, 0.5)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 0)
                            elif direction == 'on your right':
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading + 90 + 360) % 360, 0.5)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 1)
                            elif direction == 'slightly left':
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading - 30 + 360) % 360, 0.87)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 2)
                            elif direction == 'slightly right':
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading + 30 + 360) % 360, 0.87)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 3)
                            elif direction == 'ahead':
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading, 1.5)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 4)
                            elif direction == 'on your left rear':
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading - 150 + 360) % 360, 0.87)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 5)
                            elif direction == 'on your right rear':
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading + 150 + 360) % 360, 0.87)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 6)
                            else:
                                landmark_x, landmark_y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading + 180) % 360, 1.7)
                                landmark_labelx, landmark_labely = self.calculate_landmark_label(set_x, set_y, curr_heading, 7)
                            label_num = 0
                            for key, value in landmarks_dic.items():
                                if value.lower() == landmark.lower():
                                    label_num = key
                            
                            set_landmarkx_y.append((landmark_x, landmark_y))
                            set_landmark_labelx_y.append((landmark_labelx, landmark_labely))
                            set_landmark.append(label_num)
                            landmarks.append(landmark)
        
        for line in set_road_line:
            p.line([x for x, _ in line], [y for _, y in line], color="lightgray", line_width=4, line_alpha=0.8, line_color="black")

        for road in set_road:
            p.patch([x for x, _ in road], [y for _, y in road], color="lightgray")
        
        p.line(set_x, set_y, line_width=2, color="red", line_alpha=0.6)
        highlight_source = ColumnDataSource(data={'x': [x for x, _ in set_landmarkx_y], 'y': [y for _, y in set_landmarkx_y]})
        start_point = ColumnDataSource(data={'x': [self.x1], 'y': [self.y1]})
        p.scatter(x='x', y='y', size=5, color='blue', source=highlight_source, marker="circle")
        p.scatter(x='x', y='y', size=5.5, color='red', source=start_point, marker="circle")
        arrow = self.draw_arrow(set_x[-1], set_y[-1], curr_heading)
        p.add_layout(arrow)
        intersection_points = {'x': [x for x, _ in set_intersection_xy], 'y': [y for _, y in set_intersection_xy], 'labels': ["Intersection"] * len(set_intersection_xy)}
        source_intersection_points = ColumnDataSource(intersection_points)
        Range = max(max(set_y) - min(set_y), max(set_x) -min(set_x))
        if Range <= 2:
            points = {'x': [1] + [x for x, _ in set_landmark_labelx_y], 'y': [0.9] + [y for _, y in set_landmark_labelx_y], 'labels': ['start']+set_landmark}
            intersection_label = LabelSet(x='x', y='y', text='labels', source=source_intersection_points, text_align='center', text_font_size="9pt", text_color="purple", text_alpha=0.45)
        elif Range <= 12:
            points = {'x': [1] + [x for x, _ in set_landmark_labelx_y], 'y': [0.6] + [y for _, y in set_landmark_labelx_y], 'labels': ['start']+set_landmark}
            intersection_label = LabelSet(x='x', y='y', text='labels', source=source_intersection_points, text_align='center', text_font_size="9pt", text_color="purple", text_alpha=0.45)
        elif Range <= 30:
            points = {'x': [1] + [x for x, _ in set_landmark_labelx_y], 'y': [0] + [y for _, y in set_landmark_labelx_y], 'labels': ['start']+set_landmark}
            intersection_label = LabelSet(x='x', y='y', text='labels', source=source_intersection_points, text_align='center', text_font_size="8pt", text_color="purple", text_alpha=0.45)
        else:
            points = {'x': [1] + [x for x, _ in set_landmark_labelx_y], 'y': [-1] + [y for _, y in set_landmark_labelx_y], 'labels': ['start']+set_landmark}
            intersection_label = LabelSet(x='x', y='y', text='labels', source=source_intersection_points, text_align='center', text_font_size="7pt", text_color="purple", text_alpha=0.45)
 
        p.add_layout(intersection_label)
        sourcepoint = ColumnDataSource(points)
        if Range <= 12:
            points_labels = LabelSet(x='x', y='y', text='labels', source=sourcepoint, text_align='center', text_font_size="9pt")
        elif Range <= 30:
            points_labels = LabelSet(x='x', y='y', text='labels', source=sourcepoint, text_align='center', text_font_size="10pt")
        else:
            points_labels = LabelSet(x='x', y='y', text='labels', source=sourcepoint, text_align='center', text_font_size="11pt")
        p.add_layout(points_labels)

        filename=os.path.join(output_dir, filename)
        p.renderers = []
        return filename

    def draw_4way_intersection(self, x, y):
        road_vertical = [(x- self.roadWidth, y- self.roadWidth*self.elongation), (x- self.roadWidth, y+ self.roadWidth*self.elongation), (x+ self.roadWidth, y+ self.roadWidth*self.elongation), (x+ self.roadWidth, y- self.roadWidth*self.elongation)]
        road_horizontal = [(x- self.roadWidth*self.elongation, y+ self.roadWidth), (x+ self.roadWidth*self.elongation, y+ self.roadWidth), (x+ self.roadWidth*self.elongation, y- self.roadWidth), (x- self.roadWidth*self.elongation, y- self.roadWidth)]
        road_vertical_line1, road_vertical_line2 = [(x- self.roadWidth, y- self.roadWidth*self.elongation), (x- self.roadWidth, y+ self.roadWidth*self.elongation)], [(x+ self.roadWidth, y+ self.roadWidth*self.elongation), (x+ self.roadWidth, y- self.roadWidth*self.elongation)]
        road_horizontal_line1, road_horizontal_line2 = [(x- self.roadWidth*self.elongation, y+ self.roadWidth), (x+ self.roadWidth*self.elongation, y+ self.roadWidth)], [(x+ self.roadWidth*self.elongation, y- self.roadWidth), (x- self.roadWidth*self.elongation, y- self.roadWidth)]
        return road_vertical, road_horizontal, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2

    def draw_5way_intersection(self, x, y):
        road_vertical = [(x- self.roadWidth, y- self.roadWidth*self.elongation), (x- self.roadWidth, y+ self.roadWidth*self.elongation), (x+ self.roadWidth, y+ self.roadWidth*self.elongation), (x+ self.roadWidth, y- self.roadWidth*self.elongation)]
        road_horizontal = [(x- self.roadWidth*self.elongation, y+ self.roadWidth), (x+ self.roadWidth*self.elongation, y+ self.roadWidth), (x+ self.roadWidth*self.elongation, y- self.roadWidth), (x- self.roadWidth*self.elongation, y- self.roadWidth)]
        road_vertical_line1, road_vertical_line2 = [(x- self.roadWidth, y- self.roadWidth*self.elongation), (x- self.roadWidth, y+ self.roadWidth*self.elongation)], [(x+ self.roadWidth, y+ self.roadWidth*self.elongation), (x+ self.roadWidth, y- self.roadWidth*self.elongation)]
        road_horizontal_line1, road_horizontal_line2 = [(x- self.roadWidth*self.elongation, y+ self.roadWidth), (x+ self.roadWidth*self.elongation, y+ self.roadWidth)], [(x+ self.roadWidth*self.elongation, y- self.roadWidth), (x- self.roadWidth*self.elongation, y- self.roadWidth)]

        virtual_x, virtual_y = self.calculate_next_position(x, y, -135, self.roadWidth * 3)
        side_road = [(self.calculate_next_position(virtual_x, virtual_y, -45, self.roadWidth * 0.7)), (self.calculate_next_position(x, y, -45, self.roadWidth * 0.7)),
             (self.calculate_next_position(x, y, 135, self.roadWidth * 0.7)), (self.calculate_next_position(virtual_x, virtual_y, 135, self.roadWidth * 0.7))]
        side_road_line1, side_road_line2 = [(self.calculate_next_position(virtual_x, virtual_y, -45, self.roadWidth * 0.7)), (self.calculate_next_position(x, y, -45, self.roadWidth * 0.7))], [(self.calculate_next_position(x, y, 135, self.roadWidth * 0.7)), (self.calculate_next_position(virtual_x, virtual_y, 135, self.roadWidth * 0.7))]
        return road_vertical, road_horizontal, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2, side_road, side_road_line1, side_road_line2


    def normalize_angle(self, angle, reference):
        return (angle - reference) % 360

    def t_intersection_mode(self, out_headings, main_heading):
        no_normalized_angels = [angle - main_heading for angle in out_headings]
        normalized_angles = [self.normalize_angle(angle, main_heading) for angle in out_headings]
        
        zero_position = normalized_angles.index(0)
        angle_diffs = [(normalized_angles[(i + 1) % 3] - normalized_angles[i]) % 360 for i in range(3)]

        if zero_position == 0:
            if angle_diffs[0] > angle_diffs[2] and angle_diffs[0] >= 140 and angle_diffs[0] <= 220:
                if no_normalized_angels[2] < 0:
                    return 3
                elif no_normalized_angels[2] > 0 and no_normalized_angels[2] > 180:
                    return 3
                else:
                    return 1
            elif angle_diffs[2] > angle_diffs[0] and angle_diffs[2] >= 140 and angle_diffs[2] <= 220:
                if no_normalized_angels[0] < 0:
                    return 3
                elif no_normalized_angels[0] > 0 and no_normalized_angels[0] > 180:
                    return 3
                else:
                    return 1
            elif all(x >= 0 for x in no_normalized_angels) and max(no_normalized_angels) < 180:
                return 1
            elif all(x <= 0 for x in no_normalized_angels) and min(no_normalized_angels) > -180:
                return 3
            else:
                return 0
        elif zero_position == 1:
            if angle_diffs[0] > angle_diffs[1] and angle_diffs[0] >= 140 and angle_diffs[0] <= 220:
                if no_normalized_angels[1] < 0:
                    return 3
                elif no_normalized_angels[1] > 0 and no_normalized_angels[1] > 180:
                    return 3
                else:
                    return 1
            elif angle_diffs[1] > angle_diffs[0] and angle_diffs[1] >= 140 and angle_diffs[1] <= 220:
                if no_normalized_angels[0] < 0:
                    return 3
                elif no_normalized_angels[0] > 0 and no_normalized_angels[0] > 180:
                    return 3
                else:
                    return 1
            elif all(x >= 0 for x in no_normalized_angels) and max(no_normalized_angels) < 180:
                return 1
            elif all(x <= 0 for x in no_normalized_angels) and min(no_normalized_angels) > -180:
                return 3
            else:
                return 0
        else:
            if angle_diffs[2] > angle_diffs[1] and angle_diffs[2] >= 140 and angle_diffs[2] <= 220:
                if no_normalized_angels[1] < 0:
                    return 3
                elif no_normalized_angels[1] > 0 and no_normalized_angels[1] > 180:
                    return 3
                else:
                    return 1
            elif angle_diffs[1] > angle_diffs[2] and angle_diffs[1] >= 140 and angle_diffs[1] <= 220:
                if no_normalized_angels[2] < 0:
                    return 3
                elif no_normalized_angels[2] > 0 and no_normalized_angels[2] > 180:
                    return 3
                else:
                    return 1
            elif all(x >= 0 for x in no_normalized_angels) and max(no_normalized_angels) < 180:
                return 1
            elif all(x <= 0 for x in no_normalized_angels) and min(no_normalized_angels) > -180:
                return 3
            else:
                return 0

    def draw_3way_intersection(self, x, y, mode, curr_heading):
        if (mode==1 and curr_heading ==0) or (mode==0 and curr_heading==90) or (mode==3 and curr_heading==180):
            road_vertical = [(x-self.roadWidth, y-self.roadWidth*self.elongation), (x-self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y-self.roadWidth*self.elongation)]
            road_horizontal = [(x-self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth, y+self.roadWidth), (x+self.roadWidth, y-self.roadWidth), (x-self.roadWidth*self.elongation, y-self.roadWidth)]
            road_vertical_line1, road_vertical_line2 = [(x-self.roadWidth, y-self.roadWidth*self.elongation), (x-self.roadWidth, y+self.roadWidth*self.elongation)], [(x+self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y-self.roadWidth*self.elongation)]
            road_horizontal_line1, road_horizontal_line2 = [(x-self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth, y+self.roadWidth)], [(x+self.roadWidth, y-self.roadWidth), (x-self.roadWidth*self.elongation, y-self.roadWidth)]
            return road_vertical, road_horizontal, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2
        
        elif (mode==1 and curr_heading == 90) or (mode==0 and curr_heading==180) or (mode==3 and curr_heading==270):
            road_vertical = [(x-self.roadWidth, y-self.roadWidth), (x-self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y-self.roadWidth)]
            road_horizontal = [(x-self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth*self.elongation, y-self.roadWidth), (x-self.roadWidth*self.elongation, y-self.roadWidth)]
            road_vertical_line1, road_vertical_line2 = [(x-self.roadWidth, y-self.roadWidth), (x-self.roadWidth, y+self.roadWidth*self.elongation)], [(x+self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y-self.roadWidth)]
            road_horizontal_line1, road_horizontal_line2 = [(x-self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth*self.elongation, y+self.roadWidth)], [(x+self.roadWidth*self.elongation, y-self.roadWidth), (x-self.roadWidth*self.elongation, y-self.roadWidth)]
            return road_vertical, road_horizontal, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2
            
        elif (mode==1 and curr_heading == 180) or (mode==0 and curr_heading==270) or (mode==3 and curr_heading==0):
            road_vertical = [(x-self.roadWidth, y-self.roadWidth*self.elongation), (x-self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y-self.roadWidth*self.elongation)]
            road_horizontal = [(x-self.roadWidth, y+self.roadWidth), (x+self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth*self.elongation, y-self.roadWidth), (x-self.roadWidth, y-self.roadWidth)]
            road_vertical_line1, road_vertical_line2 = [(x-self.roadWidth, y-self.roadWidth*self.elongation), (x-self.roadWidth, y+self.roadWidth*self.elongation)], [(x+self.roadWidth, y+self.roadWidth*self.elongation), (x+self.roadWidth, y-self.roadWidth*self.elongation)]
            road_horizontal_line1, road_horizontal_line2 = [(x-self.roadWidth, y+self.roadWidth), (x+self.roadWidth*self.elongation, y+self.roadWidth)], [(x+self.roadWidth*self.elongation, y-self.roadWidth), (x-self.roadWidth, y-self.roadWidth)]
            return road_vertical, road_horizontal, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2
            
        elif (mode==1 and curr_heading == 270) or (mode==0 and curr_heading==0) or (mode==3 and curr_heading==90):
            road_vertical = [(x-self.roadWidth, y-self.roadWidth*self.elongation), (x-self.roadWidth, y+self.roadWidth), (x+self.roadWidth, y+self.roadWidth), (x+self.roadWidth, y-self.roadWidth*self.elongation)]
            road_horizontal = [(x-self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth*self.elongation, y-self.roadWidth), (x-self.roadWidth*self.elongation, y- self.roadWidth)]
            road_vertical_line1, road_vertical_line2 = [(x-self.roadWidth, y-self.roadWidth*self.elongation), (x-self.roadWidth, y+self.roadWidth)], [(x+self.roadWidth, y+self.roadWidth), (x+self.roadWidth, y-self.roadWidth*self.elongation)]
            road_horizontal_line1, road_horizontal_line2 = [(x-self.roadWidth*self.elongation, y+self.roadWidth), (x+self.roadWidth*self.elongation, y+self.roadWidth)], [(x+self.roadWidth*self.elongation, y-self.roadWidth), (x-self.roadWidth*self.elongation, y- self.roadWidth)]
            return road_vertical, road_horizontal, road_vertical_line1, road_vertical_line2, road_horizontal_line1, road_horizontal_line2
        else:
            raise Exception("3-way intersection heading error!")

    def draw_arrow(self, x, y, curr_heading):
        if curr_heading == 0:
            return Arrow(end=VeeHead(fill_color="red", fill_alpha=0.7, size=10, line_color="red", line_alpha=0), 
              x_start=x, y_start=y, x_end=x, y_end=y+0.4, line_width=2, line_alpha=0.6, line_color="red")
        elif curr_heading == 90:
            return Arrow(end=VeeHead(fill_color="red", fill_alpha=0.7, size=10, line_color="red", line_alpha=0), 
              x_start=x, y_start=y, x_end=x+0.4, y_end=y, line_width=2, line_alpha=0.6, line_color="red")
        elif curr_heading == 180:
            return Arrow(end=VeeHead(fill_color="red", fill_alpha=0.7, size=10, line_color="red", line_alpha=0), 
              x_start=x, y_start=y, x_end=x, y_end=y-0.4, line_width=2, line_alpha=0.6, line_color="red")
        elif curr_heading == 270:
            return Arrow(end=VeeHead(fill_color="red", fill_alpha=0.7, size=10, line_color="red", line_alpha=0), 
              x_start=x, y_start=y, x_end=x-0.4, y_end=y, line_width=2, line_alpha=0.6, line_color="red")
        else:
            raise Exception("draw map error! heading error!")

    def calculate_next_position(self, x, y, angle_deg, distance):

        angle_rad = np.radians(angle_deg)
        x2 = x + distance * np.sin(angle_rad)
        y2 = y + distance * np.cos(angle_rad)
        return x2, y2

    def calculate_road(self, curr_heading, x1_road, y1_road, x2_road, y2_road):
        if curr_heading == 0:
            road = [(x1_road-self.roadWidth, y1_road-self.roadWidth), (x2_road-self.roadWidth, y2_road+self.roadWidth), (x2_road+self.roadWidth, y2_road+self.roadWidth), (x1_road+self.roadWidth, y1_road-self.roadWidth)]
            line1, line2 = [(x1_road-self.roadWidth, y1_road-self.roadWidth), (x2_road-self.roadWidth, y2_road+self.roadWidth)], [(x2_road+self.roadWidth, y2_road+self.roadWidth), (x1_road+self.roadWidth, y1_road-self.roadWidth)]
        elif curr_heading == 90:
            road = [(x1_road-self.roadWidth, y1_road+self.roadWidth), (x2_road+self.roadWidth, y2_road+self.roadWidth), (x2_road+self.roadWidth, y2_road-self.roadWidth), (x1_road-self.roadWidth, y1_road-self.roadWidth)]
            line1, line2 = [(x1_road-self.roadWidth, y1_road+self.roadWidth), (x2_road+self.roadWidth, y2_road+self.roadWidth)], [(x2_road+self.roadWidth, y2_road-self.roadWidth), (x1_road-self.roadWidth, y1_road-self.roadWidth)]
        elif curr_heading == 180:
            road = [(x1_road+self.roadWidth, y1_road+self.roadWidth), (x2_road+self.roadWidth, y2_road-self.roadWidth), (x2_road-self.roadWidth, y2_road-self.roadWidth), (x1_road-self.roadWidth, y1_road+self.roadWidth)]
            line1, line2 = [(x1_road+self.roadWidth, y1_road+self.roadWidth), (x2_road+self.roadWidth, y2_road-self.roadWidth)], [(x2_road-self.roadWidth, y2_road-self.roadWidth), (x1_road-self.roadWidth, y1_road+self.roadWidth)]
        elif curr_heading == 270:
            road = [(x1_road+self.roadWidth, y1_road-self.roadWidth), (x2_road-self.roadWidth, y2_road-self.roadWidth), (x2_road-self.roadWidth, y2_road+self.roadWidth), (x1_road+self.roadWidth, y1_road+self.roadWidth)]
            line1, line2 = [(x1_road+self.roadWidth, y1_road-self.roadWidth), (x2_road-self.roadWidth, y2_road-self.roadWidth)], [(x2_road-self.roadWidth, y2_road+self.roadWidth), (x1_road+self.roadWidth, y1_road+self.roadWidth)]
        else:
            raise Exception("draw map error! heading error!")
        return road, line1, line2

    def calculate_landmark_label(self, set_x, set_y, curr_heading, mode):
        Range = self.nav_len
        if mode == 0:
            x, y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading - 90 + 360) % 360, 0.5)
            if Range > 2 and Range <= 10:
                if curr_heading == 0:
                    x = x - 0.1
                elif curr_heading == 90:
                    y = y + 0.1
                elif curr_heading == 180:
                    x = x + 0.1
                else:
                    y = y - 0.3
            elif Range > 10 and Range <= 30:
                if curr_heading == 0:
                    x = x - 1
                elif curr_heading == 90:
                    y = y + 1
                elif curr_heading == 180:
                    x = x + 1
                else:
                    y = y - 1.5
            elif Range > 30:
                if curr_heading == 0:
                    x = x - 1.5
                elif curr_heading == 90:
                    y = y + 1.5
                elif curr_heading == 180:
                    x = x + 1.5
                else:
                    y = y - 2
        elif mode == 1:
            x, y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading + 90 + 360) % 360, 0.5)
            if Range <= 2:
                if curr_heading == 0:
                    x = x + 0.1
                elif curr_heading == 90:
                    y = y - 0.3
                elif curr_heading == 180:
                    x = x - 0.1
                else:
                    y = y + 0.1
            elif Range > 2 and Range <= 10:
                if curr_heading == 0:
                    x = x + 0.1
                elif curr_heading == 90:
                    y = y - 0.3
                elif curr_heading == 180:
                    x = x - 0.1
                else:
                    y = y + 0.1
            elif Range > 10 and Range <= 30:
                if curr_heading == 0:
                    x = x + 1
                elif curr_heading == 90:
                    y = y - 1.5
                elif curr_heading == 180:
                    x = x - 1
                else:
                    y = y + 1
            elif Range > 30:
                if curr_heading == 0:
                    x = x + 1.5
                elif curr_heading == 90:
                    y = y - 2
                elif curr_heading == 180:
                    x = x - 1.5
                else:
                    y = y + 1.5
        elif mode == 2:
            x, y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading - 30 + 360) % 360, 0.87)
            if Range <= 2:
                if curr_heading == 0:
                    x = x - 0.1
                elif curr_heading == 90:
                    y = y + 0.3
                elif curr_heading == 180:
                    x = x + 0.1
                else:
                    y = y - 0.1
            elif Range > 2 and Range <= 10:
                if curr_heading == 0:
                    x = x - 0.1
                elif curr_heading == 90:
                    y = y + 0.1
                elif curr_heading == 180:
                    x = x + 0.1
                else:
                    y = y - 0.3
            elif Range > 10 and Range <= 30:
                if curr_heading == 0:
                    x = x - 1
                elif curr_heading == 90:
                    y = y + 1
                elif curr_heading == 180:
                    x = x + 1
                else:
                    y = y - 1.5
            elif Range > 30:
                if curr_heading == 0:
                    x = x - 1.5
                elif curr_heading == 90:
                    y = y + 1.5
                elif curr_heading == 180:
                    x = x + 1.5
                else:
                    y = y - 2
        elif mode == 3:
            x, y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading + 30 + 360) % 360, 0.87)
            if Range > 2 and Range <= 10:
                if curr_heading == 0:
                    x = x + 0.1
                elif curr_heading == 90:
                    y = y - 0.3
                elif curr_heading == 180:
                    x = x - 0.1
                else:
                    y = y + 0.1
            elif Range > 10 and Range <= 30:
                if curr_heading == 0:
                    x = x + 1
                elif curr_heading == 90:
                    y = y - 1.5
                elif curr_heading == 180:
                    x = x - 1
                else:
                    y = y + 1
            elif Range > 30:
                if curr_heading == 0:
                    x = x + 1.5
                elif curr_heading == 90:
                    y = y - 2
                elif curr_heading == 180:
                    x = x - 1.5
                else:
                    y = y + 1.5
        elif mode == 4:
            if Range <= 2:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading, 1.5)
                x = x - 0.1
            elif Range <= 10:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading, 1.7)
                x = x - 0.3
            elif Range <= 30:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading, 2)
                x = x - 0.8
            else:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading, 2.3)
                x = x - 1
        elif mode == 5:
            x, y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading - 150 + 360) % 360, 0.87)
            if Range <= 2:
                if curr_heading == 0:
                    x = x - 0.1
                elif curr_heading == 90:
                    y = y + 0.3
                elif curr_heading == 180:
                    x = x + 0.1
                else:
                    y = y - 0.1
            elif Range > 2 and Range <= 10:
                if curr_heading == 0:
                    x = x - 0.1
                elif curr_heading == 90:
                    y = y + 0.1
                elif curr_heading == 180:
                    x = x + 0.1
                else:
                    y = y - 0.3
            elif Range > 10 and Range <= 30:
                if curr_heading == 0:
                    x = x - 1
                elif curr_heading == 90:
                    y = y + 1
                elif curr_heading == 180:
                    x = x + 1
                else:
                    y = y - 1.5
            elif Range > 30:
                if curr_heading == 0:
                    x = x - 1.5
                elif curr_heading == 90:
                    y = y + 1.5
                elif curr_heading == 180:
                    x = x + 1.5
                else:
                    y = y - 2
        elif mode == 6:
            x, y = self.calculate_next_position(set_x[-1], set_y[-1], (curr_heading + 150 + 360) % 360, 0.87)
            if Range > 2 and Range <= 10:
                if curr_heading == 0:
                    x = x + 0.1
                elif curr_heading == 90:
                    y = y - 0.3
                elif curr_heading == 180:
                    x = x - 0.1
                else:
                    y = y + 0.1
            elif Range > 10 and Range <= 30:
                if curr_heading == 0:
                    x = x + 1
                elif curr_heading == 90:
                    y = y - 1.5
                elif curr_heading == 180:
                    x = x - 1
                else:
                    y = y + 1
            elif Range > 30:
                if curr_heading == 0:
                    x = x + 1.5
                elif curr_heading == 90:
                    y = y - 2
                elif curr_heading == 180:
                    x = x - 1.5
                else:
                    y = y + 1.5
        else:
            if Range <= 2:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading + 180, 1.5)
                x = x + 0.1
            elif Range <= 10:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading + 180, 1.8)
                x = x + 0.3
            elif Range <= 30:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading + 180, 2)
                x = x + 0.8
            else:
                x, y = self.calculate_next_position(set_x[-1], set_y[-1], curr_heading + 180, 2.3)
                x = x + 1
        return x, y

            