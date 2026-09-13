from dotenv import load_dotenv

# Load API keys and other environment variables
# from the .env file.
load_dotenv()


from langchain_mistralai import ChatMistralAI

from langchain.tools import tool

from langchain_core.messages import HumanMessage

from rich import print


# ============================================================
# 1. CREATE A TOOL
# ============================================================

# @tool converts this normal Python function
# into a LangChain Tool.
#
# The LLM can see this tool and decide whether
# it needs to call it.

@tool
def get_text_length(text: str) -> int:
    """
    Returns the number of characters in a given text.
    """

    # Python calculates the number of characters.
    return len(text)


# ============================================================
# 2. CREATE A TOOL LOOKUP / REGISTRY
# ============================================================

# We create a dictionary where:
#
#     Tool name → Tool object
#
# This allows us to find the correct Python tool
# when the LLM asks us to call a tool.

tools = {
    "get_text_length": get_text_length
}


# ============================================================
# 3. CREATE THE LLM
# ============================================================

# Create the Mistral chat model.

llm = ChatMistralAI(
    model="mistral-small-2506"
)


# ============================================================
# 4. BIND THE TOOL TO THE LLM
# ============================================================

# bind_tools() tells the LLM:
#
# "You are allowed to use this tool if you need it."
#
# IMPORTANT:
# bind_tools() does NOT execute the tool.
#
# It only makes the tool available to the LLM
# and gives the LLM information about:
#
# - tool name
# - tool description
# - tool input schema

llm_with_tool = llm.bind_tools([
    get_text_length
])


# ============================================================
# 5. CREATE MESSAGE HISTORY
# ============================================================

# This list stores the conversation.
#
# We will keep adding:
#
# User message
#      ↓
# AI tool-call request
#      ↓
# Tool result
#      ↓
# AI final response

message = []


# ============================================================
# 6. GET USER INPUT
# ============================================================

prompt = input("You: ")


# Convert the user's text into a LangChain HumanMessage.

query = HumanMessage(prompt)


# Add the user's message to conversation history.

message.append(query)


# ============================================================
# 7. FIRST LLM RUN
# ============================================================

# IMPORTANT:
# This is the FIRST invoke().
#
# The LLM receives the user's question.
#
# The LLM looks at the available tools and decides:
#
# "Do I need to call a tool?"
#
# If it needs the tool, the LLM DOES NOT directly execute
# our Python function.
#
# Instead, it returns a TOOL CALL request.

result = llm_with_tool.invoke(message)


# Add the AI response to the message history.
#
# This response may contain a tool_call.

message.append(result)


# ============================================================
# 8. CHECK WHETHER LLM REQUESTED A TOOL
# ============================================================

# result.tool_calls contains the tool calls
# requested by the LLM.
#
# Example:
#
# [
#     {
#         "name": "get_text_length",
#         "args": {
#             "text": "Hello world"
#         },
#         "id": "abc123",
#         "type": "tool_call"
#     }
# ]

if result.tool_calls:

    # --------------------------------------------------------
    # GET TOOL NAME
    # --------------------------------------------------------

    # Get the name of the tool requested by the LLM.

    tool_name = result.tool_calls[0]["name"]


    # --------------------------------------------------------
    # EXECUTE THE TOOL
    # --------------------------------------------------------

    # Use our tools dictionary to find the correct
    # Python function.
    #
    # Then execute it using .invoke().
    #
    # For example:
    #
    # get_text_length.invoke({
    #     "text": "Hello world"
    # })
    #
    # might return:
    #
    # 11

    tool_message = tools[tool_name].invoke(
        result.tool_calls[0]
    )


    # Add the tool result to the conversation.
    #
    # Now the message history contains:
    #
    # 1. HumanMessage
    # 2. AIMessage containing tool_call
    # 3. ToolMessage containing the tool result

    message.append(tool_message)


# ============================================================
# 9. SECOND LLM RUN
# ============================================================

# IMPORTANT:
# This is the SECOND invoke().
#
# Now the LLM receives the complete conversation:
#
# User question
#       +
# AI's previous tool call
#       +
# Actual tool result
#
# The LLM can now use the tool result
# to generate the FINAL answer.

result = llm_with_tool.invoke(message)


# ============================================================
# 10. PRINT FINAL ANSWER
# ============================================================

# result.content contains the final text generated
# by the LLM.

print(result.content)