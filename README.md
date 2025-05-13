
## 🚀 Getting Started

### 🧰 Requirements

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- Docker & Docker Compose

---

### 📥 Local Installation

```bash
git clone https://github.com/chaimajaballah0/ai-agent.git
cd ai-agent
poetry install --no-root
```

To activate poetry environment:

```bash
poetry env list
source ...
```



Create a .env file in the root

```bash
# API Keys
GEMINI_API_KEY=
SERPAPI_API_KEY=
LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=

# Variables
LLM_MODEL=
PROJECT=

# Paths
GMAIL_CREDS_FILE_PATH=.google/client_creds.json
GMAIL_TOKEN_PATH=.google/token.json

DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
DB_NAME=

```

### In order to connect to Google Service for Gmail Service you need to create a `.google` folder in the project root and add a `client_creds.json` which will store later a `token.json` file in that same folder so you wouldn't have to reauthenticate each time.

### 🐳 Using Docker

To spin up the app and database together:

```bash
docker-compose up --build
```
Or, to only run the Postgres database:

```bash
docker-compose up -d db
```

Then run the app locally:

```bash
poetry run python src/main.py
```

## 🧭 LangGraph Flow

 ```mermaid

graph TD;
        __start__(<p>__start__</p>)
        simple_or_complex(simple_or_complex)
        planning(planning)
        postprocess(postprocess)
        __end__(<p>__end__</p>)
        __start__ --> simple_or_complex;
        simple_or_complex --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc

```

I created 3 subgraphs inspired from LLMCompiler Architecture:
- Classification subgraph:  Classify the query of the user into simple or complex prompt:\
        - Simple the agent responds directly.\
        - Complex, the agent needs to access to tools ( emailing, search web, browse website
        ).\

- Plan And Execute Subgraph: In the case the query was complex, the agent needs to prepare a plan, schedule tasks, executes them and repeat until nothing can be done anymore.

- Post Processing Subgraph: In case the input was from the Plan And Execute Subgraph, then the answer of the agent needs to be structured and put nicely (summarized, organized, ..) for the end user to read.


## ⚠️ Limitations

While this project provides a modular framework for LangGraph-based agent workflows, it has the following limitations:

- The Plan And Execute subgraph is doing the tool calling properly but I still haven't managed to end the loop of replanning and scheduling.

- This leads to not being able to test the Post Processing Subgraph on the output of the previous subgraph.

- Using free-tier for the gemini api key is sometimes frustrating because the agent sometimes needs to do multiple calls per minute which leads to 500 error from the api key.

