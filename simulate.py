import random
from main import *
from utils import GPT_QA

topic_json={
  "Science": ["Physics", "Chemistry", "Biology", "Astronomy"],
  "Health & Wellness": ["Medical Information", "Nutrition", "Mental Health", "Fitness", "Gym","Yoga","Cutting"],
  "Pets & Animals": ["Pet Care", "Animal Behavior", "Wildlife Conservation"],
  "Cooking & Food": ["Recipes", "Cooking Techniques", "Food Science"],
  "Technology": ["Gadgets", "Software Development", "Cybersecurity"],
  "Arts & Humanities": ["Literature", "History", "Visual Arts", "Music"],
  "Education & Career": ["Academic Advice", "Career Guidance"],
  "Travel & Geography": ["Destination Guides", "Cultural Norms", "Outdoor Adventures"],
  "Entertainment": ["Movies & TV Shows", "Video Games", "Sports"],
  "Personal Finance": ["Budgeting", "Taxes"],
  "Social Issues & Philosophy": ["Ethics", "Social Justice"],
  "Environmental Protection": ["Climate Change", "Conservation Efforts", "Sustainable Living", "Pollution Prevention"],
  "LGBTQ+": ["Rights & Advocacy", "Health & Wellness", "History & Culture", "Community Support"]
}
user_utterence_template='''
#Task
According to provided [Conversation Context] and follow th [Instruction], play a role of a user and deliver your utterence.

#Instruction
{instruction}

#Conversation Context
{context}

#Format
Please return strictly following the format below:
{{
    "user_utterance": "Your utterance here"
}}
'''

agent_system = AgentSystem(user_name='xiaohong')

def input(self, user_utterance):

    
    index=1
    while True:
        one_turn_acn_reply_list=[]
        for reply in agent_system.handle_account_manager(user_utterance):
            print(f'acn: {reply}')
            one_turn_acn_reply_list.append(reply)
        agent_system.acn_reply_list.append(one_turn_acn_reply_list)
        agent_system.graph.visualize(f'/home/yunxshi/Data/workspace/ACN/visualize/{agent_system.user_name}/session_{session_id}/graph_{index}.png')
        agent_system.historical_graph.append(copy.deepcopy(agent_system.graph))
        agent_system.graph.clear()
        if user_utterance == 'exit':
            break
        index+=1
            

instruction_options= ['''You played user has interest in the acn's response, continue this chat.''', '''You have much interest, and want chat in depth in the current topic.''','''Give a feedback to the acn's reponse. The format should be: Give feedback: .... For example: 1.Give feedback: Reply using emoji. 2.Give feedback: Reply using a little cute girl tone. 3.Give feedback: Your reply is too long, please make it shorter. 4.Give feedback: Too redundant, please be concise but also need to provide details in key information. 5. Give feedback: The part of xxx is not good, please improve it in xxx''']
def simulate():
    name_index=random.randint(1,20)
    user_name=f'test_user_{name_index}'
    
    
    agent_system = AgentSystem(user_name=user_name)
    os.makedirs(f'/home/yunxshi/Data/workspace/ACN/visualize/{agent_system.user_name}/session_{agent_system.session_id}', exist_ok=True)
    user_profile, user_profile_string = agent_system.account_manager.get_profile(agent_system.user_name)
    agent_system.account_manager.update_historical_qa('user', f'User Profile:\n{user_profile_string}\n\n')
    
    
    num_of_turns=random.randint(1,5)
    topic=random.choice(list(topic_json.keys()))
    subtopic=random.choice(topic_json[topic])
    
    context=[]
    instruction=f'Ask a question about {topic}.{subtopic}'
    for utterence_index in range(num_of_turns):
        if len(context)==0:
            context_string='Current no conversation context, you begin the chat.'
        else:
            context_string=json.dumps(context)
        generate_utterence_prompt=user_utterence_template.format(instruction=instruction,context=context_string)
        functioncall_list,response_message=GPT_QA(generate_utterence_prompt, model_name="gpt-4o-mini", t=0.0,historical_qa=None)
        user_utterence=json.loads(response_message)['user_utterance']
        
        one_turn_acn_reply_list=[]
        for reply in agent_system.handle_account_manager(user_utterence):
            one_turn_acn_reply_list.append(reply)
        agent_system.acn_reply_list.append(one_turn_acn_reply_list)
        agent_system.graph.visualize(f'/home/yunxshi/Data/workspace/ACN/visualize/{agent_system.user_name}/session_{agent_system.session_id}/graph_{utterence_index+1}.png')
        agent_system.historical_graph.append(copy.deepcopy(agent_system.graph))
        agent_system.graph.clear()
        
        context.append({
            'role':'user',
            'content':user_utterence
        })
        context.append({
            'role':'acn',
            'content':one_turn_acn_reply_list
        })
        instruction=random.choice(instruction_options)

    with open('/home/yunxshi/Data/workspace/ACN/dataset/simulate.jsonl','a') as f:
        f.write(json.dumps(context))
        f.write('\n')
        
        
        
        
if __name__ == '__main__':
    simulate()
    
    #simulate()