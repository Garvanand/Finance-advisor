from typing import List
from haystack import Document
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from dotenv import load_dotenv
import os 
import gradio as gr
import requests
from opal import Policy, PolicyEngine
from opal.cedar.policy_engine import CedarPolicyEngine

load_dotenv()

os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

def documentize(article):
    return Document(content=article['content'], meta={'title': article['title'], 'keywords': article.get('keywords', [])})

class FinancialDataFetcher():

    def fetch_data(self, queries: List[str]) -> List[Document]:
        articles = []
        try:
            for query in queries:
                response = requests.get(FINANCIAL_API_URL, params={'query': query, 'max_results': 1})
                response.raise_for_status()
                data = response.json()
                articles.extend([documentize(article) for article in data['articles']])
        except Exception as e:
            print(e)
            print(f"Couldn't fetch articles for queries: {queries}" )
        return articles

keyword_llm = HuggingFaceTGIGenerator("mistralai/Mistral-7B-Instruct-v0.2")
keyword_llm.warm_up()

llm = HuggingFaceTGIGenerator("mistralai/Mistral-7B-Instruct-v0.2")
llm.warm_up()

keyword_prompt_template = """
Your task is to convert the following question into 3 keywords related to personal finance.
Here is an example:
question: "How can I save money for a down payment on a house?"
keywords:
save money
down payment
house
---
question: {{ question }}
keywords:
"""

prompt_template = """
Answer the question truthfully based on the given financial articles.
If the articles don't contain an answer, use your existing knowledge.

q: {{ question }}
Articles:
{% for article in articles %}
  {{article.content}}
  keywords: {{article.meta['keywords']}}
  title: {{article.meta['title']}}
{% endfor %}
"""

keyword_prompt_builder = PromptBuilder(template=keyword_prompt_template)
prompt_builder = PromptBuilder(template=prompt_template)

fetcher = FinancialDataFetcher()

pipe = Pipeline()

pipe.add_component("keyword_prompt_builder", keyword_prompt_builder)
pipe.add_component("keyword_llm", keyword_llm)
pipe.add_component("financial_data_fetcher", fetcher)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

pipe.connect("keyword_prompt_builder.prompt", "keyword_llm.prompt")
pipe.connect("keyword_llm.replies", "financial_data_fetcher.queries")
pipe.connect("financial_data_fetcher.articles", "prompt_builder.articles")
pipe.connect("prompt_builder.prompt", "llm.prompt")

admin_policy = Policy("admin_policy", rules={"admin": {"allow": {"action": ["edit", "delete"]}}})
guest_policy = Policy("guest_policy", rules={"guest": {"allow": {"action": ["view"]}}})
policy_engine = CedarPolicyEngine()
policy_engine.load_policy(admin_policy)
policy_engine.load_policy(guest_policy)

def enforce_access_control(user_role, action):
    decision = policy_engine.evaluate(user_role, action)
    return decision

USER_ROLES = {"admin", "guest"}
ACTIONS = {"view", "edit", "delete"}

def handle_request(user_role, action):
    if user_role not in USER_ROLES:
        return "Invalid user role."
    if action not in ACTIONS:
        return "Invalid action."
    decision = enforce_access_control(user_role, action)
    if decision:
        return f"Permission granted for {action} action."
    else:
        return f"Permission denied for {action} action."

iface = gr.Interface(fn=handle_request, 
                     inputs=["text", "text"], 
                     outputs="text",  
                     title="Access Control Demo",
                     description="Enter user role and action to check access control permissions.",
                     examples=[["admin", "edit"], ["guest", "delete"], ["admin", "view"]],
                     theme=gr.themes.Soft(),
                     allow_flagging="never")

iface.launch(debug=True)
