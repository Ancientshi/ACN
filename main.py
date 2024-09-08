from config import *
from AccountManager import AccountManager
from SolutionStrategist import SolutionStrategist
from InformationManager import InformationManager
from ContentCreator import ContentCreator
from Reflector_Optimizer import Reflector,Optimizer
import time
from prompt import *
from tools import return_tools,tools_description_dict
import json
from rich.console import Console
import logging
from graph import *
import copy

#create article dataset log user visualize folder
os.makedirs('article', exist_ok=True)
os.makedirs('dataset', exist_ok=True)
os.makedirs('log', exist_ok=True)
os.makedirs('user', exist_ok=True)
os.makedirs('visualize', exist_ok=True)


# VERBOSE_LEVEL = 25
VERBOSE_LEVEL = 25
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kwargs)

logging.Logger.verbose = verbose

console = Console()

class AgentSystem:
    def __init__(self, user_name):
        self.user_name = user_name
        self.set_agent()
        self.session_id=time.time()
        self.whole_doc_dict = {}
        self.whole_img_dict = {}
        
        logging.basicConfig(level=VERBOSE_LEVEL, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S.%f %Z', 
                    filename=f'log/{self.user_name}_session_{self.session_id}.log', 
                    filemode='a')  

        self.logger = logging.getLogger(__name__)
        self.logger.verbose(f'Session Begin...')
        self.graph=Graph()
        self.historical_graph=[]
        self.acn_reply_list=[]

    
    def set_agent(self):
        AccountManagerTools, SolutionStrategistTools, ReflextionTools, OptimizationTools=return_tools(tools_description_dict)
        self.reflector = Reflector(description=Reflextion_description, tools=ReflextionTools,model_name='gpt-4o-mini')
        self.optimizer = Optimizer(description=Optimization_description, tools=OptimizationTools,model_name='gpt-4o-mini')
        self.account_manager = AccountManager(description=AccountManager_description, tools=AccountManagerTools, model_name='gpt-4o-mini')
        self.solution_strategist = SolutionStrategist(description=SolutionStrategist_description, tools=SolutionStrategistTools, model_name='gpt-4o-mini')
        self.information_manager = InformationManager()
        self.content_creator = ContentCreator(description=ContentCreator_description, tools=None, model_name='gpt-4o-mini')
        
        
    def handle_content_creator(self, doc_dict, img_dict, outline, requirement, user_profile):
        external_knowledge = ""
        for index, (title, content) in enumerate(doc_dict.items()):
            external_knowledge += f'#{index}. {title}\n\n{content}\n\n'
        
        user_profile_json = json.dumps(user_profile, indent=4)
        image_sources = ''.join([f'{description}:{url}\n\n' for description, url in img_dict.items()])
        
        input_data = f'External Knowledge:\n{external_knowledge}\n\n[Image Source]:{image_sources}\n\nUser Profile:\n{user_profile_json}\n\nOutline:\n{outline}\n\nRequirement:\n{requirement}'
        functioncall_list, message = self.content_creator.forward(input_data, 0.2)
        return message
    
    def handle_solution_strategist(self, user_requirement, user_profile,stg_node_id):
        created_content_list = []

        user_content=f'User Requirement:\n{user_requirement}\n\nUser Profile:\n{user_profile}\n\n'
        functioncall_list, message = self.solution_strategist.forward(user_content, t=0.2, tool_choice=None)

        reply_dict={'reply':message}
        self.logger.verbose(f'Agent: SolutionStrategist, Function: Reply, Parameter: {reply_dict}')
        title = message.split('Title:')[1].split('Outline:')[0].strip()
        title=title.strip('"')
        outline = message.split('Outline:')[1].split('Plan:')[0].strip()
        plan = message.split('Plan:')[1].strip()
        print(plan)
        functioncall_list=[]
        for line in plan.split('\n'):
            line=line.strip()
            if 'Information Manager with query' in line:
                query=line.split('query: "')[1].split('",')[0]
                topn=int(line.split('topn: ')[1].strip('.'))
                functioncall_list.append((None,'ContactInformationManager',{'query':query,'topn':topn}))
            elif 'Content Creator' in line:
                requirement=line.split('requirement: "')[1].split('"')[0]
                functioncall_list.append((None,'ContactContentCreator',{'requirement':requirement}))
            elif 'Finalize Article' in line:
                functioncall_list.append((None,'FinalizeArticle',{'title':title}))

            
        for id, function, parameter in functioncall_list:
            verbose_text = f"Agent: SolutionStrategist, Function: {function}, Parameter: {parameter}"
            self.logger.verbose(verbose_text)
            
            if function == 'ContactInformationManager':
                query = parameter['query']
                topn = parameter['topn']
                doc_dict, img_dict = self.information_manager.forward([query], n=topn)
                
                parameter_dict ={'FindDoc':[],'FindImg':[]}
                for key,value in doc_dict.items():
                    parameter_dict['FindDoc'].append(key)
                
                for key,value in img_dict.items():
                    parameter_dict['FindImg'].append(f'{key}:{value}')
                verbose_text = f"Agent: InformationManager, Function: SearchBing, Parameter: {parameter_dict}"
                self.logger.verbose(verbose_text)
                
                self.whole_doc_dict.update(doc_dict)
                self.whole_img_dict.update(img_dict)

                self.solution_strategist.update_historical_qa(role='user', content=f'The information about {query} is already searched.')

                im_node_id=time.time()
                im_node=TreeNode(im_node_id,'Agent.InformationManager',f'The information about {query} is already searched. Searched {topn} webpages.')
                im_node.set_input(f'MessageAgent: {function}, Parameter: {parameter}')
                im_node.set_output(f'The information about {query} is already searched. Searched {topn} webpages.')
                self.graph.add_node(im_node)
                self.graph.add_edge(stg_node_id,im_node_id,f'MessageAgent: {function}, Parameter: {parameter}')
                
            
            elif function == 'ContactContentCreator':
                requirement = parameter['requirement']
                
                cc_node_id=time.time()
                cc_node=TreeNode(cc_node_id,'Agent.ContentCreator','')   
                cc_node.set_input(f'MessageAgent: {function}, Parameter: {parameter}')     
                self.graph.add_node(cc_node)
                self.graph.add_edge(stg_node_id,cc_node_id,f'MessageAgent: {function}, Parameter: {parameter}')
                
                created_content = self.handle_content_creator(self.whole_doc_dict, self.whole_img_dict, outline, requirement, user_profile)
                cc_node.set_output(created_content)
                parameter_dict={'report':created_content}
                
                verbose_text = f"Agent: ContentCreator, Generating Report, Parameter: {parameter_dict}"
                self.logger.verbose(verbose_text)
                
                created_content_list.append(created_content)
                self.solution_strategist.update_historical_qa(role='user', content=f'The content with requirement {requirement} is already created.')
                
                
                
            
            elif function == 'FinalizeArticle':
                title = parameter['title']
                article = '\n\n'.join(created_content_list)
                article_revised = '\n'.join([f'#{line}' if line.startswith('#') else line for line in article.split('\n')])
                
                os.makedirs(f'article/{self.user_name}/session_{self.session_id}', exist_ok=True)
                with open(f'article/{self.user_name}/session_{self.session_id}/{title}.md', 'w') as f:
                    f.write(f'# {title}\n\n')
                    f.write(f'''## Outline\n```markmap\n{outline}\n```\n\n''')
                    f.write(article_revised)
                
                FinalizeArticle_node_id=time.time()
                FinalizeArticle_node=TreeNode(FinalizeArticle_node_id,'Tool.FinalizeArticle','')
                FinalizeArticle_node.set_input(f'CallTool: {function}, Parameter: {parameter}')
                FinalizeArticle_node.set_output(f'The article is finalized: {title}.md')
                self.graph.add_node(FinalizeArticle_node)
                self.graph.add_edge(stg_node_id,FinalizeArticle_node_id,f'CallTool: {function}, Parameter: {parameter}')
                
                return article,outline
            

    
    def handle_account_manager(self, user_utterance):
        functioncall_list, message = self.account_manager.forward(user_utterance, t=0.2, tool_choice='required')
        self.logger.verbose(f'User Utterance: {user_utterance}')
        
        user_utterance_node_id=time.time()
        user_utterance_node=TreeNode(user_utterance_node_id,'User',self.user_name)
        user_utterance_node.set_output(user_utterance)
        self.graph.add_node(user_utterance_node)
        account_manager_node_id=time.time()
        account_manager_node=TreeNode(account_manager_node_id,'Agent.AccountManager','')
        account_manager_node.set_input(user_utterance)
        account_manager_output=''
        for id, function, parameter in functioncall_list:
            account_manager_output+=f'CallTool: {function}, Parameter: {parameter}\n'
        account_manager_node.set_output(account_manager_output)
        self.graph.add_node(account_manager_node)
        self.graph.add_edge(user_utterance_node_id,account_manager_node_id,f'User Utterance: {user_utterance}')


        for id, function, parameter in functioncall_list:
            verbose_text = f"Agent: AccountManager, Function: {function}, Parameter: {parameter}"
            self.logger.verbose(verbose_text)

            if function == 'NormalReply':
                reply = parameter['reply']
                yield reply
                self.account_manager.update_historical_qa('assistant', reply)
                
                NormalReply_node_id=time.time()
                NormalReply_node=TreeNode(NormalReply_node_id,'Tool.NormalReply','')
                NormalReply_node.set_input(f'CallTool: {function}, Parameter: {parameter}')
                NormalReply_node.set_output(reply)
                self.graph.add_node(NormalReply_node)
                self.graph.add_edge(account_manager_node_id,NormalReply_node_id,f'CallTool: {function}, Parameter: {parameter}')
                
            elif function == 'ClarifyingQuestions':
                question = parameter['question']
                yield question
                self.account_manager.update_historical_qa('assistant', question)
                
                ClarifyingQuestions_node_id=time.time()
                ClarifyingQuestions_node=TreeNode(ClarifyingQuestions_node_id,'Tool.ClarifyingQuestions','')
                ClarifyingQuestions_node.set_input(f'CallTool: {function}, Parameter: {parameter}')
                ClarifyingQuestions_node.set_output(question)
                self.graph.add_node(ClarifyingQuestions_node)
                self.graph.add_edge(account_manager_node_id,ClarifyingQuestions_node_id,f'CallTool: {function}, Parameter: {parameter}')
            
            elif function == 'ProvidingSuggestions':
                suggestion = parameter['suggestion']
                yield suggestion
                self.account_manager.update_historical_qa('assistant', suggestion)
                
                ProvidingSuggestions_node_id=time.time()
                ProvidingSuggestions_node=TreeNode(ProvidingSuggestions_node_id,'Tool.ProvidingSuggestions','')
                ProvidingSuggestions_node.set_input(f'CallTool: {function}, Parameter: {parameter}')
                ProvidingSuggestions_node.set_output(suggestion)
                self.graph.add_node(ProvidingSuggestions_node)
                self.graph.add_edge(account_manager_node_id,ProvidingSuggestions_node_id,f'CallTool: {function}, Parameter: {parameter}')
            
            elif function == 'ContactSolutionStrategist':
                stg_node_id=time.time()
                stg_node=TreeNode(stg_node_id,'Agent.SolutionStrategist','')
                stg_node.set_input(f'User Requirement: {parameter["user_requirement"]}')
                self.graph.add_node(stg_node)
        
                user_profile = self.account_manager.get_profile(username=self.user_name)
                user_requirement = parameter['user_requirement']
                article,outline = self.handle_solution_strategist(f'User Requirement:\n{user_requirement}', user_profile,stg_node_id)
                stg_node.content=f'The Solution Strategist Agent writes a outline: {outline}'
                stg_node.set_output(f'The Solution Strategist Agent writes a outline: {outline}')
                self.graph.add_edge(account_manager_node_id,stg_node_id,f'MessageAgent: {function}, Parameter: {parameter}')
                 
                yield article
                self.account_manager.update_historical_qa('assistant', f'Generated an article, the outline is {outline}, details are downloaded for you.')
                message={'User Requirement': user_requirement,'User Profile': user_profile}
                
            
            elif function == 'TrackingUserPreferences':
                description = parameter['description']
                category = parameter['category']
                self.account_manager.update_profile(username=self.user_name, description=description, category=category)
                self.account_manager.update_historical_qa('assistant', f'Already captured your preferences: "{description}" in the category of "{category}"', tool_call_id=id, name='TrackingUserPreferences')
                yield f'Already captured your preferences: "{description}" in the category of "{category}"'
                
                TrackingUserPreferences_node_id=time.time()
                TrackingUserPreferences_node=TreeNode(TrackingUserPreferences_node_id,'Tool.TrackingUserPreferences','')
                TrackingUserPreferences_node.set_input(f'CallTool: {function}, Parameter: {parameter}')
                TrackingUserPreferences_node.set_output(f'Already captured your preferences: "{description}" in the category of "{category}"')
                self.graph.add_node(TrackingUserPreferences_node)
                self.graph.add_edge(account_manager_node_id,TrackingUserPreferences_node_id,f'CallTool: {function}, Parameter: {parameter}')
            
            elif function == 'AcceptingFeedbackAndReflection':
                feedback = parameter['feedback']
                user_utterance_node.feedback=feedback
                self.RFO(feedback)
                yield 'Feedback is accepted, and we have reflected on it and do some improvement.'
                self.account_manager.update_historical_qa('assistant', 'Feedback is accepted, and we have reflected on it and do some improvement.', tool_call_id=id, name='AcceptingFeedbackAndReflection')
                
                AcceptingFeedbackAndReflection_node_id=time.time()
                AcceptingFeedbackAndReflection_node=TreeNode(AcceptingFeedbackAndReflection_node_id,'Tool.AcceptingFeedbackAndReflection','')
                AcceptingFeedbackAndReflection_node.set_input(f'CallTool: {function}, Parameter: {parameter}')
                AcceptingFeedbackAndReflection_node.set_output(f'Feedback is accepted, and we have reflected on it and do some improvement.')
                self.graph.add_node(AcceptingFeedbackAndReflection_node)
                self.graph.add_edge(account_manager_node_id,AcceptingFeedbackAndReflection_node_id,f'CallTool: {function}, Parameter: {parameter}') 
    
    
    def RFO(self,feedback,his_num=1):
        review_collection={
            'User':[],
            'Agent.AccountManager':[],
            'Agent.SolutionStrategist':[],
            'Agent.ContentCreator':[],
            'Agent.InformationManager':[],
            'Tool.NormalReply':[],
            'Tool.ClarifyingQuestions':[],
            'Tool.ProvidingSuggestions':[],
            'Tool.TrackingUserPreferences':[],
            'Tool.AcceptingFeedbackAndReflection':[],
            'Tool.FinalizeArticle':[]
        }
        current_parameter_collection={
            'User':'',
            'Agent.AccountManager':self.account_manager.description,
            'Agent.SolutionStrategist':self.solution_strategist.description,
            'Agent.ContentCreator':self.content_creator.description,
            'Agent.InformationManager':'',
            'Tool.NormalReply':tools_description_dict['NormalReply'],
            'Tool.ClarifyingQuestions':tools_description_dict['ClarifyingQuestions'],
            'Tool.ProvidingSuggestions':tools_description_dict['ProvidingSuggestions'],
            'Tool.TrackingUserPreferences':tools_description_dict['TrackingUserPreferences'],
            'Tool.AcceptingFeedbackAndReflection':tools_description_dict['AcceptingFeedbackAndReflection'],
            'Tool.FinalizeArticle':tools_description_dict['FinalizeArticle']
        }
            
        optimized_parameter_collection={
            'User':'',
            'Agent.AccountManager':'',
            'Agent.SolutionStrategist':'',
            'Agent.ContentCreator':'',
            'Agent.InformationManager':'',
            'Tool.NormalReply':'',
            'Tool.ClarifyingQuestions':'',
            'Tool.ProvidingSuggestions':'',
            'Tool.TrackingUserPreferences':'',
            'Tool.AcceptingFeedbackAndReflection':'',
            'Tool.FinalizeArticle':''
        }
        
        his_num=min(his_num,len(self.historical_graph))
        
        for graph_index in range(len(self.historical_graph)-his_num,len(self.historical_graph)):
            graph = self.historical_graph[graph_index]
            feedback_dict={}
            
            input=graph.edge_list[0][2]
            output=self.acn_reply_list[graph_index]
            
            for edge in graph.edge_list:
                call_agent_id=edge[0]
                called_agent_or_tool_id=edge[1]
                message=edge[2]
                call_agent=graph.find_node_by_id(call_agent_id)
                called_agent_or_tool=graph.find_node_by_id(called_agent_or_tool_id)
                
                if call_agent is not None and called_agent_or_tool is not None and called_agent_or_tool.type not in ['User','Agent.InformationManager']:
                    feedback_dict[called_agent_or_tool_id]=feedback
                    called_agent_or_tool.feedback=feedback
                    
                    functioncall_list,response_message=self.reflector.reflect(call_agent=call_agent,called_agent=called_agent_or_tool,message=message,input=input,output=output,called_agent_input=called_agent_or_tool.input,called_agent_output=called_agent_or_tool.output,feedback=feedback_dict[called_agent_or_tool_id],current_parameter_collection=current_parameter_collection)
                    
                    for id, function, parameter in functioncall_list[:1]:
                        relevance=parameter['Relevance']
                        review=parameter['Review']
                        down_feedback=parameter['DownFeedback']
                        if relevance == 'yes':
                            review_collection[called_agent_or_tool.type].append(review)
                            feedback=down_feedback
                        else:
                            feedback=down_feedback
                            
                    
                    self.logger.verbose(f'The RFO Optimizer optimize on {call_agent.type} -> {called_agent_or_tool.type}: {parameter}')
                                
            path=f'visualize/{self.user_name}/session_{self.session_id}/graph_{graph_index+1}_RFO.png'
            graph.visualize(path,True)

        
        with open(f'visualize/{self.user_name}/session_{self.session_id}/review_{graph_index+1}_RFO.json','w') as f:
            json.dump(review_collection,f)
         
        for node_name,review_list in review_collection.items():
            if len(review_list)>0 and node_name not in ['User','Agent.InformationManager']:
                functioncall_list,response_message=self.optimizer.optimize(node_name=node_name,review_list=review_list,current_parameter_collection=current_parameter_collection)
                for id, function, parameter in functioncall_list[:1]:
                    optimized_parameter=parameter['OptimizedParameter']
                    optimized_parameter_collection[node_name]=optimized_parameter

        with open(f'visualize/{self.user_name}/session_{self.session_id}/optimized_parameter_{graph_index+1}_RFO.json','w') as f:
            json.dump(optimized_parameter_collection,f)
        self.update_prompt(optimized_parameter_collection)
        return 'Done'
    
    def update_prompt(self,optimized_parameter_collection):
        for node_name,parameter in optimized_parameter_collection.items():
            if parameter!='':
                if node_name=='Agent.AccountManager':
                    self.account_manager.update_prompt(parameter)
                elif node_name=='Agent.SolutionStrategist':
                    self.solution_strategist.update_prompt(parameter)
                elif node_name=='Agent.ContentCreator':
                    self.content_creator.update_prompt(parameter)
                elif node_name=='Tool.NormalReply':
                    tools_description_dict['NormalReply']=parameter
                elif node_name=='Tool.ClarifyingQuestions':
                    tools_description_dict['ClarifyingQuestions']=parameter
                elif node_name=='Tool.ProvidingSuggestions':
                    tools_description_dict['ProvidingSuggestions']=parameter
                elif node_name=='Tool.TrackingUserPreferences':
                    tools_description_dict['TrackingUserPreferences']=parameter
                elif node_name=='Tool.AcceptingFeedbackAndReflection':
                    tools_description_dict['AcceptingFeedbackAndReflection']=parameter
                elif node_name=='Tool.FinalizeArticle':
                    tools_description_dict['FinalizeArticle']=parameter
                elif node_name=='User':
                    pass
                else:
                    pass
        
        AccountManagerTools, SolutionStrategistTools, ReflextionTools, OptimizationTools=return_tools(tools_description_dict)
        self.account_manager.update_tools(AccountManagerTools)
        self.solution_strategist.update_tools(SolutionStrategistTools)
        self.reflector.update_tools(ReflextionTools)
        self.optimizer.update_tools(OptimizationTools)
        return 'Done'           


          
    def run(self):
        session_id= self.session_id
        os.makedirs(f'visualize/{self.user_name}/session_{session_id}', exist_ok=True)
        
        user_profile, user_profile_string = self.account_manager.get_profile(self.user_name)
        self.account_manager.update_historical_qa('user', f'User Profile:\n{user_profile_string}\n\n')
        
        index=1
        while True:
            user_utterance = input('user: ')
            one_turn_acn_reply_list=[]
            for result in self.handle_account_manager(user_utterance):
                print(f'acn: {result}')
                one_turn_acn_reply_list.append(result)
            self.acn_reply_list.append(one_turn_acn_reply_list)
            self.graph.visualize(f'visualize/{self.user_name}/session_{session_id}/graph_{index}.png')
            self.historical_graph.append(copy.deepcopy(self.graph))
            self.graph.clear()
            if user_utterance == 'exit':
                break
            index+=1
            

if __name__ == '__main__':
    agent_system = AgentSystem(user_name='Ming')
    agent_system.run()
