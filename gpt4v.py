from openai import OpenAI
import base64
import json
client = OpenAI(api_key="sk-proj-wbtsYYRozblxiAPB1olXT3BlbkFJSjEkCseZ5fIuMUbqeOUQ")
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
def llm_getobj(name_img):
    #  Path to your image
    image_path = "data_rbg/"+name_img
    # Getting the base64 string
    base64_image = encode_image(image_path)
    img_url={"url": f"data:image/jpeg;base64,{base64_image}"}
    with open('system_prompt.txt', 'r') as file:
        system_prompt = file.read()
    response = client.chat.completions.create(
    model="gpt-4o",
    response_format={ "type": "json_object" },
    messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": img_url,
                    },
                ],
            }
        ],
            max_tokens=600,
            temperature = 0 
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

if __name__ == "__main__":
   name_img="0007.png"
   
   llm_getobj(name_img)