import openai
from time import time, sleep
import textwrap
import sys
import yaml
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

###     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)


def open_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

os.environ["OPENAI_API_KEY"] = open_file('key_openai.txt').strip()
loader = TextLoader("input.txt", encoding="utf-8")
documents = loader.load()


# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
  documents,
  embedding=OpenAIEmbeddings(),
  persist_directory='./data'
)
vectordb.persist()

qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(),
    vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=False
)

###     CHAT FUNCTIONS


def get_user_input():
    # get user input
    query = input('\n\n\nUSER:\n\n')
    
    # check if scratchpad updated, continue
    if 'DONE' in query:
        print('\n\n\nThank you for participating in this survey! Your results have been saved. Program will exit in 5 seconds.')
        sleep(5)
        exit(0)
    if query == '':
        # empty submission, probably on accident
        None
    else:
        return query


def compose_conversation(question, answer):
    # continue with composing conversation and response
    ALL_MESSAGES = []
    ALL_MESSAGES.append({question : answer})
    conversation = list()
    conversation += ALL_MESSAGES
    # conversation.append({'role': 'system', 'content': system_message})
    return conversation


def chat_print(text):
    formatted_lines = [textwrap.fill(line, width=120, initial_indent='    ', subsequent_indent='    ') for line in text.split('\n')]
    formatted_text = '\n'.join(formatted_lines)
    print('\n\n\nCHATBOT:\n\n%s' % formatted_text)

if __name__ == '__main__':
    # instantiate chatbot, variables       
    chat_history = list()
    start_time = time()
    
    # get username, start conversation
    # print('\n\n****** IMPORTANT: ******\n\nType DONE to exit\n\nSurvey Question: %s' % research_question)
    username = input('\n\n\nTo get started, please type in your name: ').strip()
    filename = f"QA_{start_time}_{username}.yaml"
    print('\n\nHello %s\n\nPlease enter your query below' % username)
    # text = f"Hello, my name is {username}."
    # conversation = compose_conversation(ALL_MESSAGES, text, system_message)
    # generate_chat_response(ALL_MESSAGES, conversation)

    while True:
        query = get_user_input()
        if not query:
            continue

        result = qa_chain({'question': query, 'chat_history': chat_history})
        conversation = compose_conversation(result['question'], result['answer'])
        save_yaml(f'chat_logs/{filename}', conversation)
        chat_print(result['answer'])
        # generate_chat_response(ALL_MESSAGES, conversation)
        save_yaml(f'chat_logs/{filename}', conversation)