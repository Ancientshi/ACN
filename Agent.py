
from utils import GPT_QA

class Agent:
    def __init__(self,description='''''',tools=None,model_name='gpt-4o-mini') -> None:
        self.description = description
        self.tools = tools
        self.model_name = model_name
        self.historical_qa = [{
            'role': 'system',
            'content': self.description
        }]
    
    def update_historical_qa(self, role='', content='',tool_call_id=None,name=None):
        assert role in ['user', 'assistant', 'tool','tool_calls']
        if role in ['tool','tool_calls']:
            assert tool_call_id != None
            assert name != None
            self.historical_qa.append({
                'role': role,
                'content': content,
                'tool_call_id': tool_call_id,
                'name': name
            })
        else:
            self.historical_qa.append({
                'role': role,
                'content': content
            })
        
    def forward(self,user_utterence,t=0.2,tool_choice='auto'):
        functioncall_list,response_message = GPT_QA(user_utterence, model_name=self.model_name,t=t, historical_qa=self.historical_qa,tools=self.tools,tool_choice=tool_choice)
        return functioncall_list,response_message
    
    def update_prompt(self,new_prompt):
        self.historical_qa[0]['content'] = new_prompt
        
    def update_tools(self,tools):
        self.tools = tools
    