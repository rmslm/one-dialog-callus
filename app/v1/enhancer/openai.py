from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.config import settings
from pydantic.v1 import BaseModel, Field


class EnhanceContextOpenAI:
    def __init__(self, *, text:list, model:str = "gpt-4o"):

        self.markdown = text[0]
        self.vision = text[1]

        self.llm = ChatOpenAI(
            temperature=0, 
            openai_api_key=settings.openai_api_key,
            model_name = model
        )

        self.json_parser = self.json_parser()

    def json_parser(self):
        # Define your desired data structure.
        class Joke(BaseModel):
            desciption: str = Field(description="desciption of the markdown section that correspond to each mapped section")
            markdown: str = Field(description="markdown section that correspond to each mapped section")

        return JsonOutputParser(pydantic_object=Joke)

    def run(self):

        messages = [
            SystemMessage(content=(
                "You are a helpful assistant that enhance the markdown text provided with the context. "
                "You write text, section by section and don't miss any. "
                "The format of the text is always paragraphs without bullet points "
                f"with this json format {self.json_parser.get_format_instructions()} with all the sections in the same list" 
            )),
            HumanMessage(
                content=(
                    f"Here is the markdown text that you have to enhance with the context provided bellow: {self.markdown}"
                    "____"
                    f"Here is the description of the markdown section by section: {self.vision}"
                    "____"
                    "Read the desciption and map the right section in the description with the right piece of markdown."
                )
            )
        ]

        response = self.llm.invoke(messages)

        try :
            output = self.json_parser.parse(response.content)
        except Exception as e :
            print("Correct json the parser")
            new_parser = OutputFixingParser.from_llm(
                parser=self.json_parser, 
                llm=self.llm
            )
            output = new_parser.parse(response.content)


        print(output)

        return output