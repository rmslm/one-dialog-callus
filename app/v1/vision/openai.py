from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

from app.v1.vision.base import BaseVison
from app.config import settings
import base64

from pydantic.v1 import BaseModel, Field

class VisionOpenAI(BaseVison):

    def __init__(self, 
                *, 
                image_path:str, 
                local_image:bool=True,
                model:str = "gpt-4o" ):

        super().__init__(image_path=image_path)

        self.local_image=local_image

        self.llm = ChatOpenAI(
            temperature=0, 
            openai_api_key=settings.openai_api_key,
            model_name = model
        )


    def run(self) -> ChatOpenAI :

        class IsForm(BaseModel):
            form  : bool = Field(
                description="If the image contains a form for a human to fill say True otherwise False"
            )

        pydantic_parser = PydanticOutputParser(pydantic_object=IsForm)
        output_format = pydantic_parser.get_format_instructions()

        messages = [
            SystemMessage(
                content="You are a helpful assistant that reads images."
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": ( 
                            "Is there a form in this image? Answer as short as possible. "
                            f"Use the following format to answer the question {output_format}"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ( 
                                f"data:image/jpeg;base64,{self.encode_image(self.image_path)}" 
                                if self.local_image 
                                else self.image_path
                            )
                        }
                    }
                ]
            )
        ]

        response = self.llm.invoke(messages)

        try :
            output = pydantic_parser.parse(response.content)
        except Exception as e :
            print("Correct fix the parser")
            new_parser = OutputFixingParser.from_llm(
                parser=pydantic_parser, 
                llm=self.llm
            )
            output = new_parser.parse(response.content)

        return output


    @staticmethod
    def encode_image(image_path:str ) -> base64:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')