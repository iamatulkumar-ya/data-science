How to download the model
Visit the Llama repository in GitHub where instructions can be found in the Llama README

    1
    Install the Llama CLI

    pip install llama-stack

In your preferred environment run the command below:
Command
Use -U option to update llama-stack if a previous version is already installed:
Command
2
Find models list

llama model list

to check the previous versions 

llama model list -sshow-all 
See latest available models by running the following command and determine the model ID you wish to download:
Command
If you want older versions of models, run the command below to show all the available Llama models:
Command
3
Select a model
llama model download --source meta --model-id  MODEL_ID
Select a desired model by running:
Command
4
Specify custom URL
Llama 2
When the script asks for your unique custom URL, please paste the URL below
URL


mine url

https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNXM2ajM2Z2N1OWJ4aHU3eXkyNWU4NGV1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMzM5NDEwNX19fV19&Signature=VK-I4eQIaNMK%7EZg-jwbj4nMP2MH-n0fqur75aWeQOfy%7E0q7gVG6cxUZxq2TiS-KV-prLlWUzDkMIft8vMMd8TfFJ6XQMuS7sP5VnZXf2FXlZEHDsMaKoqFQd46VEsfdpX9ugu7iolmvYqdqWvFUm6arbre4jPbGnkoMyBqHt3gBBIJmmA67AiHNLqkv8ERe2Q4qh6KDtFYj1Uv46X6EcW6lW0rDrh8bcHnEsPmdqnveAVbs4qUntsoQ0Mvpe7HplVYLO4hsoyiW8nCERzVG13uFuUhZzyyy7jT7g1OR7pmwBCBhuYFmX4XV37VnHXkHO0RI-iqtHt9%7EIG2wHgqT5Rg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=543161965265545


let's use public and commercial use 

Llama-2-7b

Llama-2-13b

Llama-2-70b

llama model download --source meta --model-id  MODEL_ID

llama model download --source meta --model-id  Llama-2-7b

then provide the downnload url which we got via email

13.5 GB model

To run MD5 checksums, use the following command:
llama model verify-download --model-id Llama-2-7b



torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6



    Note

    Replace llama-2-7b-chat/ with the path to your checkpoint directory and tokenizer.model with the path to your tokenizer model.
    The –nproc_per_node should be set to the MP value for the model you are using.
    Adjust the max_seq_len and max_batch_size parameters as needed.
    This example runs the example_chat_completion.py found in this repository but you can change that to a different .py file.


torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir C:\Users\Bumblebee\.llama\checkpoints \
    --tokenizer_path C:\Users\Bumblebee\.llama\checkpoints\Llama-2-7b\tokenizer.model \
    --max_seq_len 512 --max_batch_size 6