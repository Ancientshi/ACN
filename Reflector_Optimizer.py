from Agent import Agent
import json 
from AccountManager import AccountManager
from prompt import AccountManager_description,ContentCreator_description,SolutionStrategist_description
from tools import tools_description_dict

account_manager=AccountManager('Account Manager',[],'AccountManager')

def get_parameter(node, current_parameter_collection):
    node_type = node if isinstance(node, str) else node.type
    
    if node_type == 'User':
        name = node.content
        user_profile, _ = account_manager.get_profile(name)
        user_profile_json = json.dumps(user_profile)
        return 'User', user_profile_json

    type, name = node_type.split('.') if '.' in node_type else (node_type, None)
    
    if type == 'Agent':
        if name in ['AccountManager', 'ContentCreator', 'SolutionStrategist']:
            return node_type, current_parameter_collection.get(node_type)
        elif name == 'InformationManager':
            return None, None

    elif type == 'Tool':
        return node_type, current_parameter_collection.get(node_type)
    
    return None, None

class Reflector(Agent):
    def reflect(self,call_agent,called_agent,message,input,output,called_agent_input,called_agent_output,feedback,current_parameter_collection):
        '''
        call_agent 是调用方
        called_agent 是被调用方
        message 是传递给被调用方的信息
        input 是输入
        output 是输出
        '''
        node_a,call_agent_parameter=get_parameter(call_agent,current_parameter_collection)
        node_b,called_agent_parameter=get_parameter(called_agent,current_parameter_collection)
        
        content=f'''
ACN Input:
{input}

ACN Output:
{output}

Node A:
{node_a}

Node B:
{node_b}

Message:
{message}

Node B Input:
{called_agent_input}

Node B Output:
{called_agent_output}

Feedback to Node B:
{feedback}

Node B Parameter:
{called_agent_parameter}
    
        '''

        functioncall_list,response_message=self.forward(content,tool_choice='required')
        a_data={'content':content,'functioncall_list':functioncall_list,'response_message':response_message}
        with open('/home/yunxshi/Data/workspace/ACN/dataset/reflect/data.jsonl','a') as f:
            f.write(json.dumps(a_data)+'\n')
        return functioncall_list,response_message
        

class Optimizer(Agent):
    def optimize(self,node_name,review_list,current_parameter_collection):
        node_name,parameter=get_parameter(node_name,current_parameter_collection)
        review_string=[f'{index}. {review}' for index,review in enumerate(review_list)]
        review_string='\n'.join(review_string)
        content=f'''
Node Name:
{node_name}

Parameter:
```
{parameter}
```

Review:
{review_string}
        '''
        functioncall_list,response_message=self.forward(content,tool_choice='required')
        a_data={'content':content,'functioncall_list':functioncall_list,'response_message':response_message}
        with open('/home/yunxshi/Data/workspace/ACN/dataset/optimize/data.jsonl','a') as f:
            f.write(json.dumps(a_data)+'\n')
        return functioncall_list,response_message