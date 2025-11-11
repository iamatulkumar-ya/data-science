To learn the concepts for llm and langchain we are going to use open source model. For this we will be using Ollama 

Install all the requirements from `requirements_langchain.txt` file. Recommended to create individual environment for this just to avoid conflicts with other projects. Steps:

1. Create virtual environment: `py -3.12 -m venv .venv_lc`
2. Activate it: `.\.venv_lc\Scripts\activate`
3. Install all requirements `pip install -r .\requirements_langchain.txt` and set to go!!!

### OLLAMA

1. https://ollama.com/download 

2. Download the exe, once downloaded we can use gui to download the model or even command prompt.

3. use this link https://ollama.com/search to get the available model for download.

4. Let's download a llm model locally using ollama command. To download run `ollama pull <model_name>` example: `ollama pull mistral` or `ollama pull llama2`

5. It will take a while to download the model locally.

6. Once downloaded let's run that by running `ollama run llama2`

7. Availble commands: 
``` Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command
```

8. Once we run the model, we can chat over command prompt itself.