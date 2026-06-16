from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import Field
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os, json

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

class WeatherReport(BaseModel):
    city: str = Field(description="Name of the city")
    temperature: float = Field(description="Current temperature in Celsius")
    conditions: List[str] = Field(description="List of weather conditions")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in km/h")
    recommendation: str = Field(description="Recommendation for outdoor activities")
    best_travel_location: str = Field(description="Best travel location for current weather")
    way_to_change_weather: str = Field(description="Way to change current weather")

output_parser = JsonOutputParser(pydantic_object=WeatherReport)

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a weather expert. Provide current weather information in the specified JSON format. Make realistic estimations based on the season and location."""
    ),
    (
        "human",
        "{request}\n{format_instructions}"  
    )
])

chain = chat_prompt | llm | output_parser

response = chain.invoke({
    "request": "What is the weather like in Malaysia today?",
    "format_instructions": output_parser.get_format_instructions()
})

print("\nResponse:")
print(json.dumps(response, indent=4))
