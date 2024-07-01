import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os
import fitz


class PDF2Images:

    def __init__(self, pdf_path:str, batch_size:int, filename:str=None ):

        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.pdf_path = pdf_path
        self.batch_size = batch_size
        self.total_pages = fitz.open(pdf_path).page_count
        self.content_path = os.path.join(
            os.path.dirname( os.path.abspath(__file__) ), "tmp/batch_images" )

        os.makedirs( self.content_path , exist_ok=True)

        self.image_paths = []
        self.filename = filename

    
    def process_batch(self, start:int, end:int, batch_number:int):

        images = convert_from_path(
            self.pdf_path, 
            first_page=start, 
            last_page=end, 
            dpi=200
        )

        for i, image in enumerate(images):
            image = self.preprocess_image(image)
            image_file_path = f'{self.content_path}/{self.filename}_batch_{batch_number}_page_{start + i}.png'
            image.save(image_file_path)

            self.image_paths.append(image_file_path)

        del images

        return self.image_paths


    def preprocess_image(self, image):
        image_cv = np.array(image)
        if self.use_gpu:
            image_gpu = cv2.cuda_GpuMat(image_cv)
            gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)
            image_cv = gray_gpu.download()
        else:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        return Image.fromarray(image_cv)


    def run(self):

        batches = (self.total_pages + self.batch_size - 1) // self.batch_size

        for batch in range(batches):
            start_page = batch * self.batch_size + 1
            end_page = min(start_page + self.batch_size - 1, self.total_pages)

            self.process_batch(start_page, end_page, batch)

        return self.image_paths