from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

apple_resp = client.embeddings.create(
    input="I like apple juice",
    model="text-embedding-3-small"
)

orange_resp = client.embeddings.create(
    input="The weather is nice today",
    model="text-embedding-3-small"
)

# print(apple_resp.data[0].embedding )
# print(orange_resp.data[0].embedding )

print( np.sum( np.sqrt( 
    (np.array(apple_resp.data[0].embedding) -  np.array(orange_resp.data[0].embedding))**2 ) ) )


