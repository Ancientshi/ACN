from aiohttp import ClientSession
import requests
import concurrent.futures
import time
from utils import extract_page
import json

class SerperSearch:
    def __init__(self,key=""):
        self.api_key=key
        self.url = "https://google.serper.dev/search"
        self.max_chars=1000 * 5
    
    def searchserper(self,query,pagenum):
        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        

            
        # Call the API
        retry_interval_exp = 0
        while True:
            try:
                response = requests.request("POST", self.url, headers=headers, data=payload)
                response.raise_for_status()
                return response.json()
            except Exception as ex:
                if retry_interval_exp > 3:
                    return {}
                time.sleep(max(2, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1
    
    def search_content(self, query, pagenum,ignore_links=True, ignore_images=True):
        external_knowledge = {}
        all_img_dict = {}
        search_results = self.searchserper(query,pagenum)
        
        def process_result(result):
            title = result["title"]
            #如果有的result['website'],url 为 result['website']; 否则判断，如果有result['link']，url为result['link']
            # if 'website' in result.keys():
            #     url=result['website']
            if 'link' in result.keys():
                url=result['link']
            else:
                return None    
            
            page_content, img_dict = extract_page(url, ignore_links, ignore_images)
            if page_content != 'Not available':
                content_length = len(page_content)
                max_chars = min(self.max_chars, content_length)
                page_content = page_content[:max_chars]
                formatted_title = title + ':' + url
                return formatted_title, page_content, img_dict
            return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_result, result) for result in search_results['organic'][:pagenum]]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    formatted_title, page_content, img_dict=result
                    external_knowledge.update({formatted_title: page_content})
                    all_img_dict.update(img_dict)
        
        return external_knowledge,all_img_dict
    