import base64
from langchain_core.messages import SystemMessage


class BaseVisonOpenAI:
    def __init__(self, *, image_path:str, system_prompt:str=None):
        self.image_path = image_path
        self.system_prompt = system_prompt

    def run(self):
        """ Implemented in the respective class """
        raise NotImplementedError


    @staticmethod
    def dynamic_system_prompt(prompt:str = None):
        
        if prompt:
            return SystemMessage(
                content=prompt
            )
        
        return SystemMessage(
            content="You are a helpful assistant that reads images."
        )

    @staticmethod
    def encode_image(image_path:str ):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    