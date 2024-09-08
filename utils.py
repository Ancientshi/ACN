import copy
import json
import openai
import requests
import asyncio
import os
import sys
import requests
import re
import string
import numpy as np
from langchain_community.document_loaders import AsyncHtmlLoader,BSHTMLLoader
from langchain_community.document_transformers import Html2TextTransformer
import concurrent.futures
import re

  
def GPT_QA(prompt, model_name="gpt-3.5-turbo-16k", t=0.0, historical_qa=None, tools=None,tool_choice='auto'):
    url = "https://api.openai.com/v1/chat/completions"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    messages = []
    if historical_qa is not None:
        messages = copy.deepcopy(historical_qa)
    
    if prompt!=None:
        messages.append({"role": "user", "content": prompt})
    
    client = openai.OpenAI()
    if tools!=None:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=t,
            n=1,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            service_tier='default',
            max_tokens=4096,
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=t,
            n=1,
            stream=False,
            service_tier='default',
            max_tokens=4096,
        )
        
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    functioncall_list=[]
    if tool_calls:
        for tool_call in tool_calls:
                id = tool_call.id
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                functioncall_list.append([id, function_name, function_args])
    
    return functioncall_list,response_message.content

    

def extract_page(url, ignore_links=True, ignore_images=True):
    try:
        def load_docs():
            loader = AsyncHtmlLoader([url],max_wait_time=5,requests_per_second=2)
            docs=loader.load()
            if len(docs)==1:
                
                html2text = Html2TextTransformer(ignore_links=ignore_links,ignore_images=ignore_images)
                docs_transformed = html2text.transform_documents(docs)
                markdown_content=docs_transformed[0].page_content
                markdown_content,img_dict=process_markdown_images(markdown_content)
                return markdown_content,img_dict
            else:
                return "Not available", {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(load_docs)
            try:
                page_content = future.result(timeout=5)
                return page_content
            except concurrent.futures.TimeoutError:
                print("Loading the documents timed out after 5 seconds.")
                return "Not available", {}

    except Exception as e:
        print(f"An error occurred while processing the page: {e}")
        return "Not available", {}
    


def process_markdown_images(markdown_text):
    # Define a regex pattern to match markdown image links with query parameters
    pattern = r'!\[([^\]]*)\]\(([^)]*\.(jpg|jpeg|png))(\?[^)]*)?\)'
    
    # Dictionary to store image descriptions and their processed URLs
    image_dict = {}
    
    # Define a function to process each match and remove the query parameter
    def replacer(match):
        description = match.group(1)
        url = match.group(2)
        # Update the dictionary with the description and processed URL
        image_dict[description] = url
        # Return the modified markdown image syntax
        return f'![{description}]({url})'
    
    # Use re.sub to replace all occurrences and process the markdown text
    processed_text = re.sub(pattern, replacer, markdown_text)
    
    return processed_text, image_dict