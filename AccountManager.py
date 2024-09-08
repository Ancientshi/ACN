import json
from Agent import Agent
import os

class Memory:
    def __init__(self) -> None:
        self.dir = 'user'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get(self, username):
        user_profile = []
        file_path = f'{self.dir}/{username}.jsonl'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    user_profile.append(json.loads(line))
        return user_profile
    
    def add(self, username, metadata):
        file_path = f'{self.dir}/{username}.jsonl'
        with open(file_path, 'a') as f:  # Use 'a' to append to the file
            f.write(json.dumps(metadata) + '\n')

class AccountManager(Agent):
    def __init__(self,description,tools,model_name):
        super().__init__(description,tools,model_name)
        self.memory=Memory()
    
    def get_profile(self,username):
        user_profile =  self.memory.get(username=username) 
        user_profile_string=''
        for index,item in enumerate(user_profile):
            description=item['description']
            category=item['category']
            user_profile_string+=f'{index+1}. Category: {category}, Description: {description}\n'
        return user_profile,user_profile_string
    
    def update_profile(self,username,description,category):
        self.memory.add(username=username, metadata={"description":description,"category": category})
