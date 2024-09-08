import random
import os
import numpy as np
from utils import GPT_QA
import json
from tqdm import tqdm
import time

topic_json={
  "Science": ["Physics", "Chemistry", "Biology", "Astronomy","Artificial Intelligence"],
  "Health & Wellness": ["Medical Information", "Nutrition", "Mental Health", "Fitness", "Gym", "Yoga"],
  "Pets & Animals": ["Pet Care", "Animal Behavior", "Wildlife Conservation"],
  "Cooking & Food": ["Recipes", "Cooking Techniques", "Food Science"],
  "Arts & Humanities": ["Literature", "History", "Visual Arts", "Music"],
  "Education & Career": ["Overseas Study Application Consultation", "Career Guidance"],
  "Travel & Geography": ["Destination Guides", "Cultural Norms", "Outdoor Adventures"],
  "Entertainment": ["Movies & TV Shows", "Sports", "Games"],
  "Personal Finance": ["Budgeting", "Taxes"],
  "Social Issues & Philosophy": ["Ethics", "Social Justice"],
  "Environmental Protection": ["Climate Change", "Conservation Efforts", "Sustainable Living", "Pollution Prevention"],
  "LGBTQ+": ["Rights & Advocacy", "Health & Wellness", "History & Culture", "Community Support"]
}

attitude_choices=['interested', 'uninterested', 'neutral']
action_json={
  "interested":"He continue this chat content to ask a deeper or related question.",
  "uninterested":"He express the uninterested attitude and then change to another aspect question.",
  "neutral":"He responded symbolically to the chat content, but did not take the initiative or motivation to continue the conversation."
}

first_time_template="""Your task is according to the instruction to generate simulated multi-session multi-round open domain knowledge based Question-Answering dialog record dataset between a user and a AI consulting agent. These records will represent a variety of consulting scenarios, showcasing the AI agent's ability to address diverse user inquiries and provide relevant information, advice, or recommendations. The simulation aims to reflect realistic interactions, capturing how users with different attitudes (interested, uninterested, neutral) engage with the AI agent. 

The instruction is: 
{instruction}

The generated data should be in Json format, here xxx means you should fill it with your generated data, and ... means you can expand more rounds data:

[Format]
{
  "topic": "xxx",
  "subtopic": "xxx",
  "rounds": [
    {
      "user": "xxx",
      "agent": "xxx",
      "attitude": "xxx"
    },
    ...
  ]
}

Now give me the generated data for this session, make sure 1. In Json format 2. Attitude can only be one of the three: interested, uninterested, neutral. 3. No xxx and ... shown in the generated data.
"""

following_time_template="""Now continue to generate the next session data.

The instruction is: 
{instruction}

The generated data should also keep the same Json format, here xxx means you should fill it with your generated data, and ... means you can expand more rounds data

[Format]
{
  "topic": "xxx",
  "subtopic": "xxx",
  "rounds": [
    {
      "user": "xxx",
      "agent": "xxx",
      "attitude": "xxx"
    },
    ...
  ]
}


Make sure 1. In Json format 2. Attitude can only be one of the three: interested, uninterested, neutral. 3. No xxx and ... shown in the generated data.
"""

root_path='./'

def generate_one_data():
    #从3-10中随机生成一个数
    session_num=random.randint(3, 10)
    #从2-6中可放回地采样session_num个数字，存到round_num_list中
    round_num_list=  np.random.choice(range(2, 6), session_num, replace=True)

    #在每一个session, 选择一个topic下的一个subtopic
    subtopic_list=[]
    for i in range(session_num):
        topic=random.choice(list(topic_json.keys()))
        subtopic=random.choice(topic_json[topic])
        subtopic_list.append("%s -> %s"%(topic,subtopic))


    historical_qa=[]
    multi_session_data=[]
    for i in range(session_num):
        instruction=''''''
        instruction+="Session %d has %d rounds QA paris, and the whole chat topic revolves around %s.\n"%(i,round_num_list[i],subtopic_list[i])
        
        attitude_list=random.choices(attitude_choices, k=round_num_list[i]-1)
        instruction+="In the 0-th round, the user asks a question with an interested attitude about %s.\n"%(subtopic_list[i])
        for j in range(1,len(attitude_list)+1):
          attitude=attitude_list[j-1]
          action=action_json[attitude]
          instruction+="In the %s-th round, the user responses with a %s attitude toward the %s-th round's agent's reponse. %s\n"%(j,attitude_list[j-1],j-1,action)
        instruction+="\n"
        
        if i==0:
          the_template=first_time_template
        else:
          the_template=following_time_template
          
        the_template=the_template.replace("{instruction}",instruction)
        print(the_template,'-----')
        
        aa=input()
        if i==0:
          one_session_data=GPT_QA(the_template,model_name="gpt-3.5-turbo",t=0.0,historical_qa=None)
        else:
          one_session_data=GPT_QA(the_template,model_name="gpt-3.5-turbo",t=0.0,historical_qa=historical_qa)
    
        one_session_data_dict=json.loads(one_session_data)
        historical_qa.append([the_template,one_session_data])
        multi_session_data.append(one_session_data_dict)
        
    return multi_session_data

if __name__ == "__main__":
  for i in tqdm(range(0,50)):
    current_time = int(time.time())
    random.seed(current_time)
    np.random.seed(current_time)
    
    data=generate_one_data()
    data_json=json.dumps(data)
    with open(os.path.join(root_path, 'datasets/MSMTQA/test.jsonl'), 'a') as f:
      f.write(data_json+'\n')
      
    
     

'''
Your task is according to the instruction to generate simulated multi-session multi-round open domain knowledge based Question-Answering dialog record dataset between a user and a AI consulting agent. These records will represent a variety of consulting scenarios, showcasing the AI agent's ability to address diverse user inquiries and provide relevant information, advice, or recommendations. The simulation aims to reflect realistic interactions, capturing how users with different interests and attitudes engage with the AI agent.  

The instruction is: 
Session 0 has 2 rounds QA paris, and the whole chat topic revolves around Cooking & Food -> Recipes
In the 0-th round, the user asks a question with a interested attitude about Cooking & Food -> Recipes.
In the 1-th round, the user responses with a interested attitude toward the 0-th round's agent's reponse. He continue this chat content to ask a deeper or related question.
Session 1 has 4 rounds QA paris, and the whole chat topic revolves around Environmental Protection -> Climate Change
In the 0-th round, the user asks a question with a interested attitude about Environmental Protection -> Climate Change.
In the 1-th round, the user responses with a interested attitude toward the 0-th round's agent's reponse. He continue this chat content to ask a deeper or related question.
In the 2-th round, the user responses with a interested attitude toward the 1-th round's agent's reponse. He continue this chat content to ask a deeper or related question.
In the 3-th round, the user responses with a uninterested attitude toward the 2-th round's agent's reponse. He change to another aspect question.
Session 2 has 3 rounds QA paris, and the whole chat topic revolves around Social Issues & Philosophy -> Ethics
In the 0-th round, the user asks a question with a interested attitude about Social Issues & Philosophy -> Ethics.
In the 1-th round, the user responses with a uninterested attitude toward the 0-th round's agent's reponse. He change to another aspect question.
In the 2-th round, the user responses with a neutral attitude toward the 1-th round's agent's reponse. He responded symbolically to the chat content, but did not take the initiative or motivation to continue the conversation.
Session 3 has 5 rounds QA paris, and the whole chat topic revolves around Personal Finance -> Taxes
In the 0-th round, the user asks a question with a interested attitude about Personal Finance -> Taxes.
In the 1-th round, the user responses with a neutral attitude toward the 0-th round's agent's reponse. He responded symbolically to the chat content, but did not take the initiative or motivation to continue the conversation.
In the 2-th round, the user responses with a uninterested attitude toward the 1-th round's agent's reponse. He change to another aspect question.
In the 3-th round, the user responses with a uninterested attitude toward the 2-th round's agent's reponse. He change to another aspect question.
In the 4-th round, the user responses with a interested attitude toward the 3-th round's agent's reponse. He continue this chat content to ask a deeper or related question.

The generated dataset should be in Json format and the structure is as follows:
{
  "sessions": [
    {
      "topic": "xxx",
      "subtopic": "xxx",
      "rounds": [
        {
          "user": "xxx",
          "agent": "xxx",
          "attitude": "xxx"
        },
        ...
      ]
    },
    {
      "topic": "xxx",
      "subtopic": "xxx",
      "rounds": [
        {
          "user": "xxx",
          "agent": "xxx",
          "attitude": "xxx"
        },
        ...
      ]
    },
    ...
  ]
}
'''

'''
Session 0: Cooking & Food -> Recipes
Round 1

User: Can you recommend a simple pasta recipe that's quick to make?
AI Agent: Absolutely! How about a classic Aglio e Olio? You'll need spaghetti, 4 cloves of garlic thinly sliced, 1/2 cup of extra virgin olive oil, 1 teaspoon of red pepper flakes, and parsley for garnish. Cook the pasta al dente. While it's cooking, gently sauté the garlic in olive oil until golden, add red pepper flakes, then toss in the drained pasta. Garnish with parsley, and you're done!
Round 2

User: That sounds great! Do you have a suggestion for a vegan dessert?
AI Agent: Certainly! How about trying a vegan chocolate avocado mousse? You'll need 2 ripe avocados, 1/4 cup of cocoa powder, 1/4 cup of almond milk, 1/3 cup of maple syrup, and a teaspoon of vanilla extract. Blend all ingredients until smooth, chill for an hour, and enjoy!

Session 1: Environmental Protection -> Climate Change
Round 1

User: What are the main causes of climate change?
AI Agent: The main causes of climate change include the burning of fossil fuels like coal, oil, and gas, deforestation, industrial processes, and agricultural practices. These activities increase the concentration of greenhouse gases in the Earth's atmosphere, leading to global warming and climate change.
Round 2

User: How does climate change affect polar regions?
AI Agent: Climate change significantly impacts polar regions, causing melting ice sheets and glaciers, which leads to rising sea levels. It also affects wildlife, disrupting the habitat of species like polar bears and penguins. The warming can alter weather patterns and increase the frequency of extreme weather events.
Round 3

User: Can planting trees really help combat climate change?
AI Agent: Yes, planting trees can help combat climate change by absorbing carbon dioxide from the atmosphere, a major greenhouse gas. Trees also provide shade, reduce soil erosion, and support ecosystems. However, while beneficial, tree planting must be part of a broader strategy that includes reducing greenhouse gas emissions at their source.
Round 4

User: What are some solutions to mitigate climate change?
AI Agent: Solutions to mitigate climate change include transitioning to renewable energy sources like solar and wind, improving energy efficiency, adopting sustainable agriculture and forestry practices, enhancing public transportation, and reducing waste. International cooperation and policies that put a price on carbon emissions are also critical to driving widespread action.

Session 2: Social Issues & Philosophy -> Ethics
Round 1

User: What is the trolley problem in ethics?
AI Agent: The trolley problem is a thought experiment in ethics that poses a dilemma: if a runaway trolley is headed towards five people tied up on the tracks, and you have the power to switch the trolley onto another track where it would only kill one person, should you do it? It raises questions about the morality of action vs. inaction and the calculation of the greater good.
Round 2

User: How do utilitarianism and deontology differ in their approach to ethics?
AI Agent: Utilitarianism and deontology are two major ethical theories that differ significantly. Utilitarianism focuses on the consequences of actions and aims for the greatest happiness for the greatest number. Deontology, on the other hand, is concerned with the morality of actions themselves, regardless of the outcome, and upholds duties, rules, or principles as fundamental.
Round 3

User: Is it ethical to use animals for research?
AI Agent: The ethics of using animals for research is a contentious issue. Proponents argue it's necessary for medical and scientific advancement that can save human lives. Opponents contend it's cruel and unethical to subject animals to suffering for human benefit. Ethical standards in animal research often require minimizing harm and using alternatives when possible.

Session 3: Personal Finance -> Taxes
Round 1

User: How can I determine which tax bracket I'm in?
AI Agent: Your tax bracket is determined by your taxable income and filing status. To find your bracket, you'll need to look at the federal tax brackets for the current year, which are published by the IRS. Your bracket tells you the rate at which your highest dollars of income are taxed, but remember that the U.S. uses a progressive tax system, so not all of your income will be taxed at that rate.
Round 2

User: What are some common deductions I can claim to reduce my taxable income?
AI Agent: Common deductions include the standard deduction, which varies by filing status, mortgage interest, charitable contributions, medical and dental expenses above a certain threshold, and state and local taxes paid. Itemizing deductions can be beneficial if your total deductions exceed the standard deduction.
Round 3

User: Can I still contribute to an IRA for last year?
AI Agent: Yes, you typically have until the tax filing deadline in April to contribute to an IRA for the previous tax year. This can help lower your taxable income if you qualify for the deduction, depending on your income and whether you or your spouse are covered by a retirement plan at work.
Round 4

User: What's the difference between a tax credit and a tax deduction?
AI Agent: A tax credit directly reduces the amount of tax you owe, dollar for dollar. A tax deduction lowers your taxable income, which indirectly reduces the amount of tax you owe. Tax credits are generally more beneficial than deductions because they reduce your tax bill on a one-to-one basis.
Round 5

User: How does the Earned Income Tax Credit work?
AI Agent: The Earned Income Tax Credit (EITC) is a refundable tax credit for low- to moderate-income working individuals and families, particularly those with children. The amount of the EITC depends on your income, filing status, and number of children. It's designed to supplement wages and reduce poverty by encouraging and rewarding work.
'''

'''
{
  "sessions": [
    {
      "topic": "Cooking & Food",
      "subtopic": "Recipes",
      "rounds": [
        {
          "user": "Can you recommend a simple pasta recipe that's quick to make?",
          "agent": "Absolutely! How about a classic Aglio e Olio? You'll need spaghetti, 4 cloves of garlic thinly sliced, 1/2 cup of extra virgin olive oil, 1 teaspoon of red pepper flakes, and parsley for garnish. Cook the pasta al dente. While it's cooking, gently sauté the garlic in olive oil until golden, add red pepper flakes, then toss in the drained pasta. Garnish with parsley, and you're done!"
        },
        {
          "user": "That sounds great! Do you have a suggestion for a vegan dessert?",
          "agent": "Certainly! How about trying a vegan chocolate avocado mousse? You'll need 2 ripe avocados, 1/4 cup of cocoa powder, 1/4 cup of almond milk, 1/3 cup of maple syrup, and a teaspoon of vanilla extract. Blend all ingredients until smooth, chill for an hour, and enjoy!"
        }
      ]
    },
    {
      "topic": "Environmental Protection",
      "subtopic": "Climate Change",
      "rounds": [
        {
          "user": "What are the main causes of climate change?",
          "agent": "The main causes of climate change include the burning of fossil fuels like coal, oil, and gas, deforestation, industrial processes, and agricultural practices. These activities increase the concentration of greenhouse gases in the Earth's atmosphere, leading to global warming and climate change."
        },
        {
          "user": "How does climate change affect polar regions?",
          "agent": "Climate change significantly impacts polar regions, causing melting ice sheets and glaciers, which leads to rising sea levels. It also affects wildlife, disrupting the habitat of species like polar bears and penguins. The warming can alter weather patterns and increase the frequency of extreme weather events."
        },
        {
          "user": "Can planting trees really help combat climate change?",
          "agent": "Yes, planting trees can help combat climate change by absorbing carbon dioxide from the atmosphere, a major greenhouse gas. Trees also provide shade, reduce soil erosion, and support ecosystems. However, while beneficial, tree planting must be part of a broader strategy that includes reducing greenhouse gas emissions at their source."
        },
        {
          "user": "What are some solutions to mitigate climate change?",
          "agent": "Solutions to mitigate climate change include transitioning to renewable energy sources like solar and wind, improving energy efficiency, adopting sustainable agriculture and forestry practices, enhancing public transportation, and reducing waste. International cooperation and policies that put a price on carbon emissions are also critical to driving widespread action."
        }
      ]
    },
    {
      "topic": "Social Issues & Philosophy",
      "subtopic": "Ethics",
      "rounds": [
        {
          "user": "What is the trolley problem in ethics?",
          "agent": "The trolley problem is a thought experiment in ethics that poses a dilemma: if a runaway trolley is headed towards five people tied up on the tracks, and you have the power to switch the trolley onto another track where it would only kill one person, should you do it? It raises questions about the morality of action vs. inaction and the calculation of the greater good."
        },
        {
          "user": "How do utilitarianism and deontology differ in their approach to ethics?",
          "agent": "Utilitarianism and deontology are two major ethical theories that differ significantly. Utilitarianism focuses on the consequences of actions and aims for the greatest happiness for the greatest number. Deontology, on the other hand, is concerned with the morality of actions themselves, regardless of the outcome, and upholds duties, rules, or principles as fundamental."
        },
        {
          "user": "Is it ethical to use animals for research?",
          "agent": "The ethics of using animals for research is a contentious issue. Proponents argue it's necessary for medical and scientific advancement that can save human lives. Opponents contend it's cruel and unethical to subject animals to suffering for human benefit. Ethical standards in animal research often require minimizing harm and using alternatives when possible."
        }
      ]
    },
    {
      "topic": "Personal Finance",
      "subtopic": "Taxes",
      "rounds": [
        {
          "user": "How can I determine which tax bracket I'm in?",
          "agent": "Your tax bracket is determined by your taxable income and filing status. To find your bracket, you'll need to look at the federal tax brackets for the current year, which are published by the IRS. Your bracket tells you the rate at which your highest dollars of income are taxed, but remember that the U.S. uses a progressive tax system, so not all of your income will be taxed at that rate."
        },
        {
          "user": "What are some common deductions I can claim to reduce my taxable income?",
          "agent": "Common deductions include the standard deduction, which varies by filing status, mortgage interest, charitable contributions, medical and dental expenses above a certain threshold, and state and local taxes paid. Itemizing deductions can be beneficial if your total deductions exceed the standard deduction."
        },
        {
          "user": "Can I still contribute to an IRA for last year?",
          "agent": "Yes, you typically have until the tax filing deadline in April to contribute to an IRA for the previous tax year. This can help lower your taxable income if you qualify for the deduction, depending on your income and whether you or your spouse are covered by a retirement plan at work."
        },
        {
          "user": "What's the difference between a tax credit and a tax deduction?",
          "agent": "A tax credit directly reduces the amount of tax you owe, dollar for dollar. A tax deduction lowers your taxable income, which indirectly reduces the amount of tax you owe. Tax credits are generally more beneficial than deductions because they reduce your tax bill on a one-to-one basis."
        },
        {
          "user": "How does the Earned Income Tax Credit work?",
          "agent": "The Earned Income Tax Credit (EITC) is a refundable tax credit for low- to moderate-income working individuals and families, particularly those with children. The amount of the EITC depends on your income, filing status, and number of children. It's designed to supplement wages and reduce poverty by encouraging and rewarding work."
        }
      ]
    }
  ]
}
'''

'''
{
  "sessions": [
    {
      "topic": "xxx",
      "subtopic": "xxx",
      "rounds": [
        {
          "user": "xxx",
          "agent": "xxx"
        },
        ...
      ]
    },
    {
      "topic": "xxx",
      "subtopic": "xxx",
      "rounds": [
        {
          "user": "xxx",
          "agent": "xxx"
        },
        ...
      ]
    },
    ...
  ]
}
'''
