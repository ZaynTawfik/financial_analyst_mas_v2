__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import os
import streamlit as st
import warnings
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
#from utils import get_openai_api_key, get_serper_api_key

# Suppress warnings
warnings.filterwarnings('ignore')


openai_api_key = st.text_input("OpenAI API Key", type="password")

# Load API keys
#openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
#os.environ["SERPER_API_KEY"] = get_serper_api_key()
os.environ["SERPER_API_KEY"] = st.text_input("Serper API Key", type="password")

# Initialize tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define Agents
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time to identify trends.",
    backstory="Specializes in financial markets using ML for insights.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop trading strategies based on market insights.",
    backstory="Uses quantitative analysis to refine profitable strategies.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies.",
    backstory="Analyzes trade timing, price, and efficiency.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate risks associated with trading activities.",
    backstory="Scrutinizes market risks and suggests mitigation strategies.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
)

# Define Tasks
data_analysis_task = Task(
    description="Analyze market data for {stock_selection} using ML and statistical modeling.",
    expected_output="Market insights and alerts for {stock_selection}.",
    agent=data_analyst_agent,
)

strategy_development_task = Task(
    description="Develop trading strategies based on {stock_selection} insights and {risk_tolerance}.",
    expected_output="A set of trading strategies for {stock_selection}.",
    agent=trading_strategy_agent,
)

execution_planning_task = Task(
    description="Plan optimal trade execution for {stock_selection}.",
    expected_output="Execution plans for {stock_selection}.",
    agent=execution_agent,
)

risk_assessment_task = Task(
    description="Assess risks for {stock_selection} and suggest mitigation strategies.",
    expected_output="Risk analysis report for {stock_selection}.",
    agent=risk_management_agent,
)

# Create CrewAI instance
financial_trading_crew = Crew(
    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent,
    ],
    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task,
    ],
    manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    process=Process.hierarchical,
    verbose=True,
)

# 🎯 Streamlit UI
st.title("📈 AI-Powered Financial Trading Analysis")
st.write("Enter your trading preferences and let AI analyze the market.")

# Input fields
stock_selection = st.text_input("Stock Symbol (e.g., AAPL, TSLA)", "AAPL")
initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)
risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
trading_strategy_preference = st.selectbox("Trading Strategy", ["Day Trading", "Swing Trading", "Long-Term"])

# Run analysis button
if st.button("Run Analysis 🚀"):
    with st.spinner("AI is analyzing market trends..."):
        inputs = {
            "stock_selection": stock_selection,
            "initial_capital": str(initial_capital),
            "risk_tolerance": risk_tolerance,
            "trading_strategy_preference": trading_strategy_preference,
        }

        # Run CrewAI process
        result = financial_trading_crew.kickoff(inputs=inputs)

        # Display results
        st.markdown(f"### 📊 Trading Analysis Results for {stock_selection}")
        st.markdown(result)
