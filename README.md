# OmniQA

Feed virtually any type of document to a large language model and get an index in return. You can then ask question to your index and the large language model will answer with high accuracy in a few instants.

* Models supported : ChatGPT, Davinci, Curie (pretty easy to add more)

## Getting started
### Installation
* `python -m pip install -r requirements.txt`
* Add your OpenAI token inside './API_KEY.txt'
* `python OmniQA.py --help`
### How to use
* Add several documents to a new index:
    * `python OmniQA.py --create_index --index_name="mathematics" --new_docs="list_of_import_path.txt" --import_type="listfile" --extra_doc="first_import"`
* Add one more document to your index:
    * `python OmniQA.py --index_name="mathematics" --new_docs="path/to/one_pdf" --import_type="onefile"`
* Ask questions:
    * `python OmniQA.py --index_name="mathematics"`
    * Then simply enter your questions.

# FAQ
* can I search multiple index at the same time?
    * I'm pretty sure faiss and llama_index allow to do this but have not implemented it yet. The code currently includes some of that but is not functional and to be ignored.
* Are you open to contributions?
    * Definitely yes, just open an issue. Even for typos and chitchat.
* How large are the index?
    * Extremely lightweight, a few megabytes for tens of large pdfs in my case. If you have large amount of data you can quickly tweak the faiss index to suit your needs, it's incredibly fast and supports many optimization including larger than ram search.

# TODO
## high priority
* when adding files to the index, the parsing should be parallelized
* query multiple indexes
* parser: anki
## Not urgent
* parser: youtube (exists already on llama_hub i think)
* parser: url (on llama_hub)
* handle query transformations (link)[https://gpt-index.readthedocs.io/en/latest/how_to/query_transformations.html]
* upload it to pypi

### Similar projects
* [paper-qa](https://github.com/whitead/paper-qa/)
* [dr-doc-search](https://github.com/namuan/dr-doc-search)
* [DocsGPT](https://github.com/arc53/docsgpt/)

### Credits to awesome libs
* [LLama_Index (formerly GPT_index)](https://github.com/jerryjliu/gpt_index) is used for most of the heavy lifting
* [langchain](https://langchain.readthedocs.io/en/latest/index.html) is building the future
* [faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)
* [prompt_toolkit](https://pypi.org/project/prompt-toolkit/)
