# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

# load_dotenv() loads variables from the .env file
# into the environment.
#
# For example, your .env might contain:
#
# MISTRAL_API_KEY=your_api_key
# TAVILY_API_KEY=your_api_key
#
# This allows us to use API keys without writing
# them directly inside the Python code.

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# IMPORT TAVILY SEARCH TOOL
# --------------------------------------------------

# TavilySearchResults is a LangChain tool used
# to search the internet and return search results.
#
# We will use this tool to search for:
# "Latest AI news of 2026"

from langchain_community.tools.tavily_search import TavilySearchResults


# --------------------------------------------------
# IMPORT MISTRAL LLM
# --------------------------------------------------

# ChatMistralAI allows us to use a Mistral
# language model through LangChain.
#
# The LLM will take the search results
# and summarize them.

from langchain_mistralai import ChatMistralAI


# --------------------------------------------------
# IMPORT PROMPT TEMPLATE
# --------------------------------------------------

# ChatPromptTemplate is used to create
# a reusable prompt for the LLM.

from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# IMPORT OUTPUT PARSER
# --------------------------------------------------

# StrOutputParser converts the LLM response
# into a normal Python string.

from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# CREATE TAVILY SEARCH TOOL
# --------------------------------------------------

# Create a Tavily search tool.
#
# max_results=5 means:
# Tavily should return up to 5 search results.
#
# NOTE:
# The current LangChain parameter is typically
# max_results, not max_result.

search_tool = TavilySearchResults(max_results=5)


# --------------------------------------------------
# CREATE MISTRAL LLM
# --------------------------------------------------

# Create the Mistral language model.
#
# model="mistral-small-2506"
# tells LangChain which Mistral model to use.

llm = ChatMistralAI(
    model="mistral-small-2506"
)


# --------------------------------------------------
# CREATE PROMPT
# --------------------------------------------------

# ChatPromptTemplate.from_template()
# creates a reusable prompt template.
#
# {news} is a placeholder.
# Later, we will replace {news} with the
# actual Tavily search results.

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.

    Summarize the following news into clear bullet points.

    {news}
    """
)


# --------------------------------------------------
# CREATE LANGCHAIN CHAIN
# --------------------------------------------------

# The | symbol is called the LangChain
# Expression Language (LCEL) pipe operator.
#
# Data flows from left → right.
#
# Flow:
#
# prompt
#   ↓
# llm
#   ↓
# StrOutputParser
#
# So:
#
# 1. Prompt receives the news
# 2. Prompt is sent to Mistral
# 3. Mistral generates a response
# 4. StrOutputParser converts the response
#    into a normal string

chain = prompt | llm | StrOutputParser()


# --------------------------------------------------
# SEARCH FOR NEWS
# --------------------------------------------------

# Call the Tavily search tool.
#
# Tavily searches the internet for:
#
# "Latest AI news of 2026"
#
# The returned search results are stored
# inside news_result.

news_result = search_tool.invoke(
    "Latest AI news of 2026"
)


# --------------------------------------------------
# SEND NEWS TO THE LLM CHAIN
# --------------------------------------------------

# Now we send the Tavily search results
# to our prompt.
#
# {news} in the prompt will be replaced
# with the value of news_result.
#
# Then:
#
# news_result
#      ↓
# prompt
#      ↓
# Mistral LLM
#      ↓
# StrOutputParser
#      ↓
# result

result = chain.invoke({
    "news": news_result
})


# --------------------------------------------------
# PRINT FINAL RESULT
# --------------------------------------------------

# Print the summarized news.

print(result)


# --------------------------------------------------
# TOOL DESCRIPTION
# --------------------------------------------------

# .description gives information about
# what the Tavily search tool does.
#
# This description helps LangChain/AI agents
# understand the purpose of the tool.

print(search_tool.description)


# --------------------------------------------------
# TOOL NAME
# --------------------------------------------------

# .name gives the name of the LangChain tool.

print(search_tool.name)


# --------------------------------------------------
# TOOL ARGUMENTS
# --------------------------------------------------

# .args tells us what input arguments
# the Tavily tool accepts.
#
# It shows the input schema of the tool.

print(search_tool.args)