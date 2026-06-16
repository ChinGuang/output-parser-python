from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
    api_key = os.getenv("GEMINI_API_KEY")
)

output_parser = CommaSeparatedListOutputParser()
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{request} \n {format_instructions}")
    ]
)

chain = chat_prompt | llm |output_parser

ai_msg = chain.invoke({
    "request": "Give me 5 characteristics of a cat",
    "format_instructions": output_parser.get_format_instructions()
})

print(output_parser.get_format_instructions())
print("\n")
print(ai_msg)

