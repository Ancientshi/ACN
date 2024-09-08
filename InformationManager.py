from Agent import Agent
from Search import SerperSearch as BingSearch
import traceback

class InformationManager:
    def __init__(self) -> None:
        search_api_kay = os.environ["SEARCH_API_KEY"]
        self.bingsearch_tool=BingSearch(search_api_kay)
    
    def forward(self,queries,n=2):
        if isinstance(queries,str):
            queries=[queries]
            
        whole_relevant_doc_dict,whole_img_dict=self.bingsearch(queries,n)
        return whole_relevant_doc_dict, whole_img_dict

            
        
    def bingsearch(self,queries,n=2):
        whole_relevant_doc_dict = {}
        whole_img_dict = {}
        for query in queries:
            try:
                title_content_obj,img_dict=self.bingsearch_tool.search_content(query,n,ignore_images=False)
                # print(f'Before filtering length: {len(title_content_obj)}')
                # title_content_obj_filtered=filter.bge_filter(query,title_content_obj,0.3,'')
                # print(f'After filtering length: {len(title_content_obj_filtered)}')   
                whole_relevant_doc_dict.update(title_content_obj)
                whole_img_dict.update(img_dict)
            except Exception as exc:
                print(f'Query {query} generated an exception: {exc}')
                traceback.print_exc()
        print('Length of whole_relevant_doc_dict:', len(whole_relevant_doc_dict))
        return whole_relevant_doc_dict,whole_img_dict
    
#main

if __name__ == "__main__":
    queries=['Top 10 museums in Syd.']
    im=InformationManager()
    whole_relevant_doc_dict, whole_img_dict =im.forward(queries)
    index=0
    for title, content in whole_relevant_doc_dict.items():
        markdown_content = f"# {title}\n\n{content}"
        
        # Save as .md file
        with open(f'{index}.md', 'w') as f:
            f.write(markdown_content)
        
        index += 1
