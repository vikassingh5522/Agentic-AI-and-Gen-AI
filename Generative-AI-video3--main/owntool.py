from langchain.tools import tool


# @tool is a LangChain decorator.
# It converts the normal Python function below
# into a LangChain Tool object.
@tool
def get_greeting(name: str) -> str:
    """
    Generate a greeting message for a user.
    
    This docstring becomes the description of the tool.
    The LLM can use this description to understand
    what this tool does.
    """

    # The function takes the user's name
    # and returns a greeting message.
    return f"Hello {name}, Welcome to the AI world"


# --------------------------------------------------
# CALLING THE LANGCHAIN TOOL
# --------------------------------------------------

# .invoke() is used to execute a LangChain tool.
#
# The tool expects an input dictionary because
# the function has an argument called "name".
#
# {"name": "akarsh"}
#        ↓
# name = "akarsh"
#
# The function returns:
# "Hello akarsh, Welcome to the AI world"

result = get_greeting.invoke({
    "name": "akarsh"
})

print(result)


# --------------------------------------------------
# TOOL NAME
# --------------------------------------------------

# .name gives the name of the tool.
#
# It comes from the Python function name:
# def get_greeting(...)
#
# Output:
# get_greeting

print(get_greeting.name)


# --------------------------------------------------
# TOOL DESCRIPTION
# --------------------------------------------------

# .description gives the description of the tool.
#
# It comes from the function's docstring:
#
# """Generate a greeting message for a user"""
#
# Output will be something similar to:
# Generate a greeting message for a user

print(get_greeting.description)


# --------------------------------------------------
# TOOL ARGUMENTS / INPUT SCHEMA
# --------------------------------------------------

# .args tells us what input arguments
# the tool expects.
#
# Because our function has:
#
# name: str
#
# LangChain creates an input schema for "name".
#
# It will show information similar to:
# {
#     'name': {
#         'title': 'Name',
#         'type': 'string'
#     }
# }

print(get_greeting.args)