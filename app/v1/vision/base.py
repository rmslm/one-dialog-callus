
class BaseVison:
    def __init__(self, image_path:str):
        self.image_path = image_path

        

    def run(self):
        """ Implemented in the respective class """
        raise NotImplementedError


    