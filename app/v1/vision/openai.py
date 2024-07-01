from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

from app.v1.vision.base import BaseVisonOpenAI
from app.config import settings

from pydantic.v1 import BaseModel, Field


class VisionContextOpenAI(BaseVisonOpenAI):

    def __init__(self, 
                *, 
                image_path:str, 
                local_image:bool=True,
                system_prompt:str= None,
                model:str = "gpt-4o" ):

        super().__init__(image_path=image_path, system_prompt=system_prompt)

        self.local_image=local_image

        self.llm = ChatOpenAI(
            temperature=0, 
            openai_api_key=settings.openai_api_key,
            model_name = model
        )

        self.form_parser = self.is_form_parser()


    def is_form_parser(self):
        class IsForm(BaseModel):
            form  : bool = Field(
                description=(
                    "If this page contains a form for a human to fill say True otherwise False "
                    "if the image contains only text answer with False"
                )
            )

        return PydanticOutputParser(pydantic_object=IsForm)
    


    def chat_messages(self, user_prompt:str ):

        messages = [
            self.dynamic_system_prompt(prompt=self.system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"{user_prompt}"
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

        return messages



    def is_form(self) -> BaseModel :

        user_prompt = (
            "Is there a form in this image? Answer as short as possible. "
            f"Use the following format to answer the question {self.form_parser.get_format_instructions()}"
        )

        response = self.llm.invoke(self.chat_messages(user_prompt=user_prompt))

        try :
            output = self.form_parser.parse(response.content)
        except Exception as e :
            print("Correct fix the parser")
            new_parser = OutputFixingParser.from_llm(
                parser=self.form_parser, 
                llm=self.llm
            )
            output = new_parser.parse(response.content)


        return output


    def extract_context_from_pdf(self):

        user_prompt = (
            "Describe section by section the context of this pdf. Your answer should be formated in paragraphs without bullet point or titles."
            # f"Use the following format to answer the question {self.form_parser.get_format_instructions()}"
        )

        response = self.llm.invoke(self.chat_messages(user_prompt=user_prompt))
 
        return response


    def run(self) -> ChatOpenAI :

        is_form = self.is_form()

        print(is_form)

        r = self.extract_context_from_pdf()

        return r.content



    