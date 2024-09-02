from zhipuai import ZhipuAI


# add yout api_key here
client = ZhipuAI(api_key="your api key")

def call_response(prompt):
    response = client.chat.completions.create(
            model="glm-4", 
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
    answer = response.choices[0].message.content
    return answer





