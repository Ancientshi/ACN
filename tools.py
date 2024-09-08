
tools_description_dict = {
    'NormalReply': "Normally chat and respond to user utterance.",
    'ClarifyingQuestions':"When necessary if user says vague questions or feedbacks, guiding users to clarify their questions or feedbacks and provide additional information based on your understanding of their potential needs.",
    'ProvidingSuggestions':"After solving users' request, offering additional tailored suggestions or opinions to guiding user engaged in the conversation and enhance user experience.",
    'ContactSolutionStrategist':"Automatically recognize users' multimedia information retrieval and generation request, trigger contacting with the Solution Strategist when appropriate.",
    "AcceptingFeedbackAndReflection": "Automatically recognize and receive user's feedback and feelings of your provided service.",
    'TrackingUserPreferences':"Try you best and most sensitivity to track and remember user's related information. Generate a short description to recognizing user’s basic information, such as occupation, age, name, and so on. Also track the interest preferences according to the user's newest utterence. Remember only analyze one newest user's message, not tracing historical chat. Besides, need to acclocate a suitable category to the short description.",
    'ContactInformationManager':"Give query to the Information Manager agent, he can find useful multimodal information to you. For a query, the Information Manager agent can find a number of top ranking websites and will go through the web information. Default as 2.",
    'ContactContentCreator':"Give requirement to the Content Creator agent, he can generate a detailed report and adhere to the requirement. The requirement should clearly state the task he need to finish, the topic he need to write, and any other necessary details. You can let the Content Creator agent write only a section of the final article, or the whole article.",
    'FinalizeArticle': "When necessary information is collected and all related content is generated, everything is well prepared, then Finalize the article.",
    "Reflect": "Judge whether [Feedback] is relevant to [Node B]. If yes, generate a review comment to [Parameter]. Then give the downstream feedback to ACN's lattering process. If no, leave review black, and give the downstream feedback to ACN's lattering process.",
    "Optimize": "Optimize the [Parameter] for the node named [Node Name] based on the provided [Reviews]. These reviews highlight specific areas where the node's performance can be improved. Consider the feedback carefully and adjust the [Parameter] to enhance the effectiveness and efficiency of this node within the Agent Collaboration Network. The new [Parameter] should address the identified weaknesses and align with the overall goals of the ACN."
}

def return_tools(tools_description_dict):
    AccountManagerTools = [
            {
                "type": "function",
                "function": {
                    "name": "NormalReply",
                    "description": tools_description_dict['NormalReply'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reply": {
                                "type": "string",
                                "description": "The reply message.",
                            },
                        },
                        "required": ["reply"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ClarifyingQuestions",
                    "description": tools_description_dict['ClarifyingQuestions'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The clarifying questions to ask the user.",
                            },
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ProvidingSuggestions",
                    "description": tools_description_dict['ProvidingSuggestions'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "suggestion": {
                                "type": "string",
                                "description": "Suggestion opinion or questions providing to the user making them engaged in this conversation.",
                            },
                        },
                        "required": ["suggestion"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ContactSolutionStrategist",
                    "description": tools_description_dict['ContactSolutionStrategist'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_requirement": {
                                "type": "string",
                                "description": "Delivery user's requirement, often in detailed personalized multimedai information retrieval and generation request, to the Solution Strategist agent for further processing.",
                            },
                        },
                        "required": ["user_requirement"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "AcceptingFeedbackAndReflection",
                    "description": tools_description_dict['AcceptingFeedbackAndReflection'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feedback": {
                                "type": "string",
                                "description": "The feedback from the user.",
                            },
                        },
                        "required": ["feedback"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "TrackingUserPreferences",
                    "description": tools_description_dict['TrackingUserPreferences'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "A short description about the user's basic information, preference or unpreference.",
                            },
                            "category": {
                                "type": "string",
                                "description": "The category that the short description belongs to. Such as hobbies, basic info, liked movies, disliked food, etc.",
                            },
                        },
                        "required": ["description","category"],
                    },
                },
            },
        ]


    SolutionStrategistTools = [
            {
                "type": "function",
                "function": {
                    "name": "ContactInformationManager",
                    "description": tools_description_dict['ContactInformationManager'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The generated search-friendly query."
                            },
                            "topn": {
                                "type": "integer",
                                "description": "The number of pages searched by one query. Default as 2."
                            },
                        },
                        "required": ["query","topn"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ContactContentCreator",
                    "description": tools_description_dict['ContactContentCreator'],
                    "parameters":{
                        "type": "object",
                        "properties": {
                            "requirement": {
                                "type": "string",
                                "description": "The generated personalized report."
                            }
                        },
                        "required": ["requirement"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "FinalizeArticle",
                    "description": tools_description_dict['FinalizeArticle'],
                    "parameters":{
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The title of the finalized article."
                            }
                        },
                        "required": ["title"]
                    }
                }
            }
        ]





    ReflextionTools = [
            {
                "type": "function",
                "function": {
                    "name": "Reflect",
                    "description": tools_description_dict['Reflect'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Relevance": {
                                "type": "string",
                                "description": "Indicates whether the feedback is relevant to Node B's dupty.",
                                "enum": ["yes", "no"]
                            },
                            "Review": {
                                "type": "string",
                                "description": "If Relevance is yes, just generate improvement suggestion to Node B (No explaination). If Relevance is no, return 'None'."
                            },
                            "DownFeedback": {
                                "type": "string",
                                "description": "The downstream feedback to the lattering process that Node B message or call."
                            }
                        },
                        "required": ["Relevance","Review","DownFeedback"]
                    }
                }
            }
        ]

    OptimizationTools=[
                {
                "type": "function",
                "function": {
                    "name": "Optimize",
                    "description": tools_description_dict['Optimize'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "OptimizedParameter": {
                                "type": "string",
                                "description": "The optimized parameter."
                            }
                        },
                        "required": ["OptimizedParameter"]
                    }
                }
            }
    ]
    
    return AccountManagerTools, SolutionStrategistTools, ReflextionTools, OptimizationTools