import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, map_lims, step_size, max_iter):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_lims = map_lims
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [self.start]
    
    def get_random_node(self):
        x = random.uniform(self.map_lims[0, 0], self.map_lims[0, 1])
        y = random.uniform(self.map_lims[1, 0], self.map_lims[1, 1])
        return Node(x, y)
    
    def get_nearest_node(self, random_node):
        nearest_node = self.nodes[0]
        min_dist = self.distance(random_node, nearest_node)
        for node in self.nodes:
            dist = self.distance(random_node, node)
            if dist < min_dist:
                nearest_node = node
                min_dist = dist
        return nearest_node
    
    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    
    def steer(self, from_node, to_node):
        direction = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * np.cos(direction)
        new_y = from_node.y + self.step_size * np.sin(direction)
        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        return new_node
    
    def is_goal_reached(self, node):
        return self.distance(node, self.goal) < self.step_size
    
    def get_path(self, node):
        path = []
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    
    def plan(self):
        for _ in range(self.max_iter):
            random_node = self.get_random_node()
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node)
            self.nodes.append(new_node)
            
            if self.is_goal_reached(new_node):
                return np.array(self.get_path(new_node))
        
        print("No path found within the maximum iterations.")
        return None