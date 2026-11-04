from openai import OpenAI
from dotenv import load_dotenv
import os 
import json 
from utilities.utilities import log
from tools import TOOLS, FUNCTION_NAME_TO_FUNCTION_MAP
from tools import run_tool
from types import SimpleNamespace

load_dotenv()

api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")

client_options = {"api_key": api_key}

base_url = os.getenv("BASE_URL")
if base_url:
    client_options["base_url"] = base_url

client = OpenAI(**client_options)
messages = []

messages.append(
    {
        "role":"system",
        "content" : "You are a helpful assistant. When a tool returns information, "
                    "use that information to answer the user directly. Do not say the "
                    "information is unavailable when the tool returned a result."
    }
)

def stream_chat(history):
    stream = client.chat.completions.create(
                model = model,
                messages=history,
                tools=TOOLS,
                stream=True
    )

    full_response=""
    calls={}
    label_printed = False

    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 1.Content Streaming :Do we have content from LLM
            if delta.content :
                if not label_printed:
                    print("Assistant: ", end="" , flush=True)
                    label_printed=True
                
                print(delta.content , end="", flush=True)
                full_response+=delta.content

            #2. Tool calling 

            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index , 
                    SimpleNamespace(id = None , function = SimpleNamespace(name = "" , arguments="") ),
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name+=tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments+= tc.function.arguments

    except KeyboardInterrupt:
        print("{stopped by you}")
    if label_printed:
        print() #Moves to the next line after assistant response

    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])

    return full_response, calls_list

def build_toolcall_payload(tool_calls):
    """
    It converts our collected toolcalls into a dictionary format
    that the Api expects inside an assistant message
    """

    payload = []

    for tool_call in tool_calls:
        tool_call_dict ={
            "id":tool_call.id,
            "type":"function",
            "function":{
                "name":tool_call.function.name,
                "arguments":tool_call.function.arguments
            }
        }
        payload.append(tool_call_dict)

    return payload

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break

    messages.append(
            {
                "role":"user",
                "content":user_input
            }
    )
    # Max number of time it call call tool (safety limit)
    MAX_RANGE =10

    for i in range(MAX_RANGE):

        full_response , tool_calls = stream_chat(messages)

        msg = full_response
        
        if not tool_calls :
            break

        messages.append({
            "role":"assistant",
            "content":full_response,
            "tool_calls": build_toolcall_payload(tool_calls)
        })

        log(f"Step {i+1} : model requested {len(tool_calls)} tool calls" )
        

        for tool_call in tool_calls:

            res = run_tool(tool_call)


            messages.append({
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content": str(res)
            })
    else: 
        # we ran out of the steps 
        fallback = "sorry we could not finish this within the step limit"

        msg.append({
            "role":"assistance",
            "content": fallback
        })

        print("Assistant" + fallback)
        continue


    messages.append({
        "role":"assistant",
        "content":full_response
    })


