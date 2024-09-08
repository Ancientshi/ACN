import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

class TreeNode:
    def __init__(self, id, type, content):
        self.id = id
        self.type = type
        self.content = content
        self.input = None
        self.output = None
        self.feedback = ''
    
    def set_input(self, input):
        self.input = input
    
    def set_output(self, output):    
        self.output = output

class Graph:
    def __init__(self):
        self.node_list = []
        self.edge_list = []

    def add_node(self, node):
        self.node_list.append(node)

    def add_edge(self, from_node_id, to_node_id, edge_value):
        self.edge_list.append((from_node_id, to_node_id, edge_value))
    
    def find_node_by_id(self, id):
        for node in self.node_list:
            if node.id == id:
                return node
        return None
    
    def set_feedback(self, node_id, feedback):
        node = self.find_node_by_id(node_id)
        if node is not None:
            node.feedback += f'Feedback: {feedback}\n'
        else:
            print(f'Node with ID {node_id} not found')
    
    
    def clear(self):
        self.node_list = []
        self.edge_list = []

    def visualize(self, path,feedback=False):
        G = nx.DiGraph()

        # Add nodes with labels
        for node in self.node_list:
            if node.content is None:
                node_content='None'
            elif len(node.content)>20:
                node_content=node.content[:20]+'...'
            else:
                node_content=node.content
            
            if feedback:
                if len(node.feedback)>20:
                    feedback_content=node.feedback[:20]+'...'
                else:
                    feedback_content=node.feedback
                
                node_content+=f'\nFeedback: {feedback_content}'
                
            if node.type == 'User':
                G.add_node(node.id, label=f'{node.type}\n{node_content}',subset_key=0)
            elif node.type == 'Agent.AccountManager':
                G.add_node(node.id, label=f'{node.type}\n{node_content}',subset_key=1)
            elif node.type == 'Agent.SolutionStrategist':
                G.add_node(node.id, label=f'{node.type}\n{node_content}',subset_key=2)  
            elif node.type == 'Agent.ContentCreator':
                G.add_node(node.id, label=f'{node.type}\n{node_content}',subset_key=3)
            elif node.type == 'Agent.InformationManager':
                G.add_node(node.id, label=f'{node.type}\n{node_content}',subset_key=3)
            elif 'Tool' in node.type:
                G.add_node(node.id, label=f'{node.type}\n{node_content}',subset_key=4)

    
        # Add edges with a default numeric weight
        index=1
        for from_node_id, to_node_id, edge_value in self.edge_list:
            G.add_edge(from_node_id, to_node_id, weight=1.0, label=f'{index}@'+edge_value)  # Use 1.0 as a default weight
            index+=1
            
        pos = nx.multipartite_layout(G,subset_key='subset_key',scale=10)  # Increase k to spread out nodes

        # Draw nodes with labels
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=2000, node_color='lightblue', font_size=4, font_weight='bold', arrows=True)



        # Draw edge labels (string weights) with horizontal rotation
        edge_labels = nx.get_edge_attributes(G, 'label')

        for key in edge_labels:
            value=edge_labels[key]
            #最多20个字符，超过的补...
            if len(value)>20:
                value=value[:20]+'...'
            if len(value)>20:
                new_value=''
                for i in range(len(value)//20+1):
                    new_value+=value[i*20:(i+1)*20]+'\n'
                edge_labels[key]=new_value
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, rotate=False, font_size=4)  # Set rotate to False to keep labels horizontal

        # Save the plot to a file
        plt.savefig(path,dpi=600)
        plt.close()  # Ensure the plot is properly closed
        
    
                    