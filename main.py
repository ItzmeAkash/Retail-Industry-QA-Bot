from langchain.llms import google_palm
from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
from few_shots import few_shots
 
#Creating The Object
def get_few_show_db_chain():
    llm = GooglePalm(google_api_key="",temperature=0.1)


    #create the db object

    db_user = 'root'
    db_password = 'admin'
    db_host = '0.tcp.in.ngrok.io'
    df_port = "12704"
    db_name = 'atliq_tshirts'

    db = SQLDatabase.from_uri(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{df_port}/{db_name}')
    
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM=L6-v2')
    
    to_vectorize = [" ".join(example.values) for example in few_shots]
    
    vectorstore = Chroma.from_texts(to_vectorize,embeddings,metadatas=few_shots)
    
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    
    example_prompt =  PromptTemplate(
    input_variables=["Question", "SQLQuery","SQLResult","Answer",],
    template = "\nQuestion: {Question}\n SQLQuery: {SQLQuery}\n SQLResult: {SQLResult}\n Answer: {Answer} "
     )
    fewshotprompt= FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt = example_prompt,
    prefix=_mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input","table_info","top_k"]#These variables are used in the prefix and suffix

    )
    chain = SQLDatabaseChain.from_llm(llm,db,verbose=True,prompt=fewshotprompt)
    return chain




if __name__ == "__main__":
    chain = get_few_show_db_chain()
    print(chain.run("How many total t shirts are left in total in stock?"))