import openai
import time
import json


def initVar():
    global OAI_key
    global OAI

    try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file.")
        exit()

    class OAI:
        key = data["keys"][0]["OAI_key"]
        model = data["OAI_data"][0]["model"]
        prompt = data["OAI_data"][0]["prompt"]
        temperature = data["OAI_data"][0]["temperature"]
        max_tokens = data["OAI_data"][0]["max_tokens"]
        top_p = data["OAI_data"][0]["top_p"]
        frequency_penalty = data["OAI_data"][0]["frequency_penalty"]
        presence_penalty = data["OAI_data"][0]["presence_penalty"]





def llm(message):

    openai.api_key = OAI.key
    start_sequence = " #########"
    response = openai.Completion.create(
      model= OAI.model,
      prompt= OAI.prompt + "\n\n#########\n" + message + "\n#########\n",
      temperature = OAI.temperature,
      max_tokens = OAI.max_tokens,
      top_p = OAI.top_p,
      frequency_penalty = OAI.frequency_penalty,
      presence_penalty = OAI.presence_penalty
    )

    json_object = json.loads(str(response))
    return(json_object['choices'][0]['text'])

def readInput():
    i = input("OAI Question: ")
    print (llm(i))

if __name__ == "__main__":
    initVar()
    print("\n\Running!\n\n")
    while True:
        # read_chat()
        readInput()
        print("\n\nReset!\n\n")
        time.sleep(2)
