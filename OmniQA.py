import time
import logging
from pathlib import Path
import os
from typing import Dict
import code
import json
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from tqdm import tqdm
import fire
import re
from llama_index import (
        MockLLMPredictor,
        MockEmbedding,
        GPTListIndex,
        GPTFaissIndex,
        GPTSimpleVectorIndex,
        LLMPredictor,
        QuestionAnswerPrompt,
        RefinePrompt,
        PromptHelper,
        )
from llama_index.embeddings.openai import (
        OpenAIEmbedding, OpenAIEmbeddingMode, OpenAIEmbeddingModelType)
from llama_index.composability import ComposableGraph
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from langchain import OpenAI
import faiss

from utils.prompts import PromptClass
from utils.parser import ParserClass


def pl(text, *args, **kwargs):
    "simply print and log"
    text = str(text)
    log.info("PRINTED: " + text)
    tqdm.write(text, *args, **kwargs)


class OmniQA:
    """
    OmniQA is a python tool that can ingest textual data, embed it using
    large language models by OpenAI then make it easy to ask question on the
    documents.
    """
    def __init__(
            self,
            index_name="default.index",
            new_docs=None,
            import_type="onefile",
            create_index=False,
            top_k=3,
            extra_docid=None,
            prompt_name="french_qa",
            prompt_model_name="gpt-3.5-turbo",
            yes=False,
            debug=False,
            ):
        """
        Parameters
        ----------
        index_name: str, default: "default.index"
            Name of the index from the directory './Indexes' to use.
            If 'create_index': then this index will be created.
            Globs can be used but only for relative paths (ex: ./DB/**/*pdf)
            and only if 'create_index' is False.

        create_index: bool, default: False
            If True, will create a new index with the name from 'index_name'
            in the directory './Indexes/'.

        new_docs: str, default: None
            If string: is the path to a file to add to the index.
            Globs are supported for relatives paths (ex: ../*.pdf).
            Documents currently supported are :
                pdf files (.pdf)
                markdown files (.md)
                text files (.txt)
            Many more format can be added easily thanks to llama_index and
            llama_hub.
            Custom format can be added in the folder 'utils/parserClass'.

        import_type: str, default 'onefile'
            String to specify how "new_docs" should be interpreted.
            * If 'onefile': the parser will determine the type of the file
              supported by default. Globs can still be used if they point to individual files.
            * If 'listfile': new_docs is considered a path to a document containing
              one path per line. Globing is supported with relative paths.
              Lines begining with # are ignored.
            * If any other value, will be passed to ParserClass to be used
              by custom parsers.

        top_k: int, default: 3
            Number of chunks to give the LLM when looking for answer.
            This will drastically increase the token cost of each question
            but can give more meaningful answer if used well.
            You can change the top_k without reloading the index by
            using '/top_k=5' in the prompt.

        extra_docid: str, default: None
            When adding new files to the index, this string will be added
            before the default doc_id. This can be used to have some context
            on the source documents used by the prompt model when it answers.
            It also appears in the source of each answer.

        prompt_name: str, default: "french_qa"
            The name of the prompt that is loaded as attribute of PromptClass
            in the file 'utils/prompts.py'. This allows to add your
            own custom prompts.

        prompt_model_name: str, default: "gpt-3.5-turbo"
            The name of the model used to answer the question. Supported
            currently are 'gpt-3.5-turbo' (ChatGPT),
            'text-davinci-003' and 'text-curie-001'
            You can change the model by using the command
            '/model=davinci' directly in the prompt

        yes: bool, default: False
            If True, will skip instead of asking for confirmation for some
            steps. Will not skip the confirmation to ask for something too
            expansive.

        debug: bool, default: False
            if True, instead of catching most exceptions, they will be
            raised. Useful when debugging with pdb
        """
        # checking arguments validity
        # path handling
        self.ind_dir = Path("./Indexes/")
        self.ind_dir.mkdir(exist_ok=True)
        if not index_name.endswith(".index"):
            index_name += ".index"
        self.index_path = self.ind_dir / index_name
        pl("List of index currently found:")
        for ind in sorted([x for x in self.ind_dir.iterdir()]):
            pl(f"    * {ind}")
        pl("")

        self.import_type = import_type
        self.debug = debug

        # which prompt to use
        self.prompt_name = prompt_name
        prompts = PromptClass()
        assert hasattr(prompts, prompt_name), "Invalid prompt name"
        prompt = getattr(prompts, prompt_name)
        self.prompt = QuestionAnswerPrompt(prompt)

        # --yes to bypass confirmation
        assert isinstance(yes, bool), "Wrong type of --yes"
        self.yes = yes

        # name of the model to use for questions
        assert prompt_model_name.lower() in [
                "gpt-3.5-turbo",
                "text-curie-001",
                "text-davinci-003"], (
            "Invalid 'model' value")
        self.prompt_model_name = prompt_model_name.lower()

        # optional extra docid for files
        if extra_docid is not None:
            assert new_docs is not None, (
                "Must specify files to add if extra_docid is set")
            self.extra_docid = extra_docid

        if create_index:
            assert new_docs is not None, (
                "Must specify files to add to index")
            assert not self.index_path.exists(), "Index already exists."

        # cannot import files if loading a Composite index
        if "*" in str(self.index_path):
            assert new_docs is None, (
                "Cannot import documents when loading several indices")

        # warn user that
        self.nd_ispath = True
        if new_docs is not None and (import_type not in ["onefile", "list"]):
            # this is to avoid having FileNotFoundError for custom
            # parsers.
            pl("Custom import_type specified, new_docs will not be "
               "interpreted as a path to a file that must exist.")
            self.nd_ispath = False

        # if importing from file: parse it here and store
        # it in self.new_docs
        if new_docs is not None and import_type == "list":
            pl(f"Will try to import documents from list at '{new_docs}'")
            assert isinstance(new_docs, str), "Wrong type for new_docs"
            assert Path(new_docs).exists(), (
                f"'{new_docs}' file does not exist")
            file_list = Path(new_docs).read_text().strip().split("\n")
            pl("\nList of documents to import that are mentionned "
               f"in '{new_docs}'")
            to_add = []
            for fi in file_list:
                fi = fi.strip()
                if fi.startswith("#") or fi == "":
                    continue
                pl(f"    * {fi}")
                to_add.append(fi)
            if not self.yes:
                ans = input("\n\nIs that okay? (y/n)")
                if ans not in ["y", "yes"]:
                    pl("Quitting.")
                    raise SystemExit()
            if len(to_add) == 0:
                raise Exception("Empty list of files to import!")
            new_docs = to_add
        if isinstance(new_docs, str):
            new_docs = [new_docs]
        self.new_docs = new_docs

        pl("Loading API key.")
        credential_file = "./API_KEY.txt"
        self.openai_api_key = str(Path(credential_file).read_text())
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        # used to display the price of each query
        self.mock_llm_predictor = MockLLMPredictor(max_tokens=256)
        self.mock_embed_model = MockEmbedding(embed_dim=1536)

        # specifying LLM to use when prompting + its price
        self.model_info = {
                "gpt-3.5-turbo": {
                    "shortname": "chatgpt",
                    "price": 0.002,
                    "refine": RefinePrompt.from_langchain_prompt(
                        getattr(prompts, f"{prompt_name}_chat_refine")),
                    "max_tokens": 4096,  # actually chatgpt can take more
                    # but this is also used to limit the completion size
                    },
                "text-davinci-003": {
                    "shortname": "davinci",
                    "price": 0.02,
                    "refine": RefinePrompt(
                        getattr(prompts, f"{prompt_name}_default_refine")),
                    "max_tokens": 4096,
                    },
                "text-curie-001": {
                    "shortname": "curie",
                    "price": 0.002,
                    "refine": RefinePrompt(
                        getattr(prompts, f"{prompt_name}_default_refine")),
                    "max_tokens": 2048,
                    },
                }
        self.model_price = self.model_info[self.prompt_model_name]["price"]
        self.shortnames = [x["shortname"] for x in self.model_info.values()]

        self.openai_embedder = OpenAIEmbedding(
                mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
                model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
                )

        # specify number of results to fetch from index
        assert isinstance(top_k, int) and top_k >= 1, "Invalid type or value of top_k"
        self.top_k = top_k

        # wether to use multiline input by default
        self.multiline = False

        # loading parser class
        self.parser = ParserClass(pl)

        # reloading previous prompts
        self.prev_questions = []
        try:
            pl("Loading previous prompts from file.")
            pp_file = Path("previous_questions.json")
            assert pp_file.exists(), (
                "Cache file does not exist. Creating it now.")
            pp_list = json.load(pp_file.open("r"))
            assert isinstance(pp_list, list), "Invalid cache type"
            for i, pp in enumerate(pp_list):
                assert isinstance(pp, dict), "Invalid item in cache"
                assert "prompt" in pp, "Invalid item in cache"
            for pp in pp_list:
                if "timestamp" not in pp:
                    pp["timestamp"] = 0
                if pp not in self.prev_questions:
                    self.prev_questions.append(pp)
            self.prev_questions = sorted(
                    self.prev_questions,
                    key=lambda x: x["timestamp"],
                    )
        except Exception as err:
            pl(f"Exception when loading previous prompts : '{err}'")

        # init
        if create_index:
            self.add_docs_to_index(create_index=True)
        else:
            self.load_index()
            if self.new_docs:
                self.add_docs_to_index(create_index=False)

        # main loop
        self.main_loop()

    def load_new_docs(self):
        """
        Loads the documents to add then parse them.
        """
        assert isinstance(self.new_docs, list), "Invalid list to add."
        pl("Loading files to index.")

        # handling glob and folders
        to_remove = []
        for path in self.new_docs:
            if "*" in path:
                self.new_docs.extend([str(x)
                                         for x in Path(".").rglob(str(path))])
                to_remove.append(path)
                pl(f"Considering '{path}' as a glob")
        for path in self.new_docs:
            if path.lower().endswith("/"):
                pl("Detected folder, iterating with pathlib.")
                to_remove.append(path)
                self.new_docs.extend([x for x in Path(path).iterdir()])

        self.new_docs = sorted([x
                                   for x in self.new_docs
                                   if x not in to_remove])
        pl("Will try to index the following files:")
        for path in self.new_docs:
            pl(f"      * {path}")

        if not self.yes:
            ans = input("\n\nIs that okay? (y/n)")
            if ans not in ["y", "yes"]:
                pl("Quitting.")
                raise SystemExit()

        self.new_documents = []
        failed = []
        for path in tqdm(self.new_docs, desc="Loading docs"):
            if self.nd_ispath:
                assert Path(path).exists(), f"File '{path}' not found."

            res = self.parser.dispatch(
                path=path,
                extra_docid=self.extra_docid,
                import_type=self.import_type,
                )
            if res is not None:
                self.new_documents.append(res)
            else:
                failed.append(path)

        pl(f"Number of documents parsed: '{len(self.new_documents)}'")
        pl(f"Number of documents failed: '{len(failed)}'")
        for fai in failed:
            self.new_docs.remove(fai)

        if len(self.new_documents) == 0:
            raise Exception(
                "No documents were parsed without exception. Quitting.")

    def load_index(self):
        """
        Load the index whose name is specified by the user.
        """
        if "*" not in str(self.index_path):
            assert Path(self.index_path).exists(), "Index does not exist"
            assert Path(str(self.index_path) + ".faiss").exists(), (
                "Faiss store does not exist")

            self.index = GPTFaissIndex.load_from_disk(
                    self.index_path,
                    faiss_index_save_path=str(self.index_path) + ".faiss",
                    prompt_helper=PromptHelper(
                        max_input_size=4096,
                        num_output=512,
                        max_chunk_overlap=100)
                    )
            pl(f"\nLoaded index '{self.index_path}' from file.")

            # self.query_mode = "default"
            # self.query_configs = [{
            #         "index_struct_type": "dict",
            #         "query_mode": "default",
            #         }]

        else:
            pl("\nLoading all indexes at once.")
            self.query_configs = []

            index_paths = [str(x)
                           for x in Path(".").rglob(str(self.index_path))]
            if len(index_paths) == 1:
                self.index_path = index_paths[0]
                return self.load_index()  # recursive call

            raise NotImplementedError()
            # not yet implemented but llama index probably supports it
            self.index_list = []
            for index_path in index_paths:
                pl(f"Loading index '{index_path}'")
                new_index = GPTFaissIndex.load_from_disk(
                        index_path,
                        faiss_index_save_path=str(index_path) + ".faiss",
                        )
                self.index_list.append(new_index)
                self.query_configs.append({
                        "index_struct_type": "dict",
                        "query_mode": "default",
                        })
            self.index_gptlist = GPTListIndex(self.index_list)
            self.index = ComposableGraph.build_from_index(self.index_gptlist)
            self.query_mode = "recursive"

    def add_docs_to_index(self, create_index):
        """
        Add each document to the index.

        The prices are calculated beforehand and the user needs to confirm.
        """
        self.load_new_docs()

        # compute prices
        pl("\n\n\n PRICE TO INDEX THOSE FILES:")
        total = 0
        self.mock_index = GPTSimpleVectorIndex(
                [],
                llm_predictor=self.mock_llm_predictor,
                embed_model=self.mock_embed_model,
                )
        for i, doc in enumerate(tqdm(self.new_documents, desc="Pricing")):
            self.mock_index.update(doc)
            price_tkn = self.mock_embed_model.last_token_usage
            total += price_tkn
            fname = Path(self.new_docs[i]).name
            price_dol = f"{price_tkn / 1000 * 0.0004:.2f}"
            formatted_price = f"${price_dol} for '{fname}'"
            pl(formatted_price)

        total_dol = total / 1000 * 0.0004
        formatted_price = (
                f"\n\n'Total price: Token Used: {total} "
                f"Cost: ${total_dol:.2f}")
        pl(formatted_price)
        if not (self.yes and total_dol < 10):
            ans = input("\n\nAre you sure? (y/n)")
            if ans not in ["y", "yes"]:
                pl("Quitting.")
                raise SystemExit()

        if create_index:
            pl("Creating index.")

            faiss_d = 1536
            faiss_index = faiss.IndexFlatL2(faiss_d)
            # not yet working:
            # faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(faiss_d))
            self.index = GPTFaissIndex(
                    [],
                    faiss_index=faiss_index,
                    embed_model=self.openai_embedder,
                    )

        assert hasattr(self, "index"), "Index has not yet been loaded!"

        for i, doc in enumerate(tqdm(self.new_documents,
                                     desc="Indexing documents")):
            try:
                # wait at least some time between documents
                t = time.time()
                self.index.insert(doc)
                time.sleep(1)
                t = time.time() - t
                time.sleep(max(10 - t, 0))
            except Exception as err:
                pl(f"Exception '{err}' when adding '{self.new_docs[i]} to "
                   f"'{self.index_path}'. Waiting 60s then retrying")
                if self.debug:
                    pl("Waiting.")
                    raise
                time.sleep(60)

                pl("Retrying.")
                try:
                    self.index.insert(doc)
                except Exception as err:
                    pl(f"Exception AGAIN '{err}' when adding "
                       f"'{self.new_docs[i]} to "
                       f"'{self.index_path}'. "
                       "Waiting 5 minutes then retrying.")
                    if self.debug:
                        pl("Waiting.")
                        raise
                    time.sleep(60*5)

                    pl("Retrying.")
                    try:
                        self.index.insert(doc)
                    except Exception as err:
                        pl(f"Exception 3rd time '{err}' when adding "
                           f"'{self.new_docs[i]} to "
                           f"'{self.index_path}'. Skipping this document.")
                        if self.debug:
                            raise
                            pl("Waiting.")
                    continue

            self.index.save_to_disk(
                    self.index_path,
                    faiss_index_save_path=str(self.index_path) + ".faiss",
                    )
            pl(f"Added {self.new_docs[i]} to '{self.index_path}'")

    def prompt_user(self, q):
        """
        Ask the question to the user.

        Accepts multiple prompt commands:
            /top_k=3 to change the top_k value.
            /model=davinci to switch model
            /debug to open a console.
            /multiline to write your question over multiple lines.
        """
        prompt_commands = ["/multiline", "/debug", "/top_k="]
        prompt_commands.extend([f"/model={short}" for short in self.shortnames])
        autocomplete = WordCompleter(
                prompt_commands + [
                    x["prompt"]
                    for x in self.prev_questions
                    ], match_middle=True, ignore_case=True)
        try:
            try:
                if self.multiline:
                    pl("Multiline mode activated. Use ctrl+D to send.")
                ans = prompt(q,
                             completer=autocomplete,
                             vi_mode=True,
                             multiline=self.multiline)
            except (KeyboardInterrupt, EOFError):
                if self.multiline:
                    pass
                else:
                    raise

            # quit if needed
            if ans.strip() in ["quit", "Q", "q"]:
                pl("Quitting.")
                raise SystemExit()

            # auto remove duplicate "slash" (i.e. //) before prompts commands
            for pc in prompt_commands:
                if f"/{pc}" in ans:
                    ans = ans.replace(f"/{pc}", f"{pc}")

            # retry if user entered multiple commands
            if len([pc for pc in prompt_commands if pc in ans]) not in [0, 1]:
                pl("You can use at most 1 prompt command in a given query")
                return self.prompt_user(q)

            # parse prompt commands
            if "/top_k=" in ans:
                try:
                    prev = self.top_k
                    self.top_k = int(re.search(r"/top_k=(\d+)", ans).group(1))
                    ans = re.sub(r"/top_k=(\d+)", "", ans)
                    pl(f"Changed top_k from '{prev}' to '{self.top_k}'")
                except Exception as err:
                    pl(f"Error when changing top_k: '{err}'")
                    if self.debug:
                        raise
                    return self.prompt_user(q)

            if "/debug" in ans:
                pl("Entering debug mode.")
                breakpoint()
                pl("Restarting prompt.")
                return self.prompt_user(q)

            if "/model=" in ans:
                new_model = re.search(r"/model=(\w+)", ans).group(1)
                assert new_model in self.shortnames, (
                    f"model name not part of '{self.shortnames}'")
                self.switch_model(new_model)
                return self.prompt_user(q)

            if "/multiline" in ans:
                if self.multiline is False:
                    self.multiline = True
                    pl("Multiline turned on.")
                else:
                    self.multiline = False
                    pl("Multiline turned off.")
                return self.prompt_user(q)
        except (KeyboardInterrupt, EOFError):
            raise SystemExit()

        return ans

    def switch_model(self, newmodel):
        """
        change model used for question answering
        """
        prev = self.prompt_model_name
        self.prompt_model_name = [x
                for x, y in self.model_info.items()
                if self.model_info[x]["shortname"] == newmodel][0]
        self.model_price = self.model_info[self.prompt_model_name]["price"]
        pl(f"Switch from model '{prev}' to '{self.prompt_model_name}'")

    def main_loop(self, query=None):
        """
        Main loop that waits for the user to enter a question, then calculates
        the price of the question, ask for confirmation, send the query,
        show the answer.
        """
        while True:
            user_question = self.prompt_user(
                    "\n\nWhat is your question? (Q to quit)\n")

            if user_question.strip() == "":
                pl("No input.")
                continue

            if len(
                    [x
                     for x in self.prev_questions
                     if x["prompt"].strip() == user_question.strip()
                     ]) == 0:
                self.prev_questions.append(
                        {
                            "prompt": user_question,
                            "timestamp": int(time.time()),
                            "prompt_name": self.prompt_name,
                            "model_name": self.prompt_model_name,
                            })
            self.prev_questions = sorted(
                    self.prev_questions,
                    key=lambda x: x["timestamp"],
                    )
            json.dump(self.prev_questions, Path(
                "previous_questions.json").open("w"), indent=4)

            log.info(f"\nQuestion: {user_question}")

            # compute price before running
            if hasattr(self, "index_list"):  # using composite index
                raise NotImplementedError()
                price_tkn = 0
                for ind in self.index_list:
                    ind.query(
                            user_question,
                            text_qa_template=self.prompt,
                            refine_template=self.model_info[
                                self.prompt_model_name]["refine"],
                            llm_predictor=self.mock_llm_predictor,
                            embed_model=self.mock_embed_model,
                            similarity_top_k=self.top_k,
                            #response_mode="compact",
                            #optimizer=SentenceEmbeddingOptimizer(threshold_cutoff=0.5),
                            )
                    price_tkn += self.mock_llm_predictor.last_token_usage
            else:  # for regular faiss index
                self.index.query(
                        user_question,
                        text_qa_template=self.prompt,
                        refine_template=self.model_info[
                            self.prompt_model_name]["refine"],
                        llm_predictor=self.mock_llm_predictor,
                        embed_model=self.mock_embed_model,
                        similarity_top_k=self.top_k,
                        #response_mode="compact",
                        #optimizer=SentenceEmbeddingOptimizer(threshold_cutoff=0.1),
                        )
                price_tkn = self.mock_llm_predictor.last_token_usage

            total_price = price_tkn / 1000 * self.model_price
            pl(f"\nTokens Used: {price_tkn} Cost: "
               f"${total_price:.4f}")

            if total_price >= 0.3:
                ans = input(f"\n\nHigh cost question (${total_price:.2f}>30cts"
                            "), are you sure? (y/n)")
                if ans not in ["y", "yes"]:
                    pl("Cancelled, returning to prompt.")
                    continue

            max_tkn = self.model_info[self.prompt_model_name]["max_tokens"]

            self.openai_llm = LLMPredictor(
                    llm=OpenAI(
                        temperature=0,
                        model_name=self.prompt_model_name,
                        max_tokens=max(500, max_tkn - 1 - price_tkn),
                        openai_api_key=self.openai_api_key,
                        )
                    )

            pl("Querying...")
            if hasattr(self, "index_list"):  # using composeability
                raise NotImplementedError()
                response = self.index.query(
                        user_question,
                        text_qa_template=self.prompt,
                        refine_template=self.model_info[
                            self.prompt_model_name]["refine"],
                        query_configs=self.query_configs,
                        similarity_top_k=self.top_k,
                        #response_mode="compact",
                        #optimizer=SentenceEmbeddingOptimizer(threshold_cutoff=0.5),
                        )
            else:
                response = self.index.query(
                        user_question,
                        text_qa_template=self.prompt,
                        refine_template=self.model_info[
                            self.prompt_model_name]["refine"],
                        llm_predictor=self.openai_llm,
                        embed_model=self.openai_embedder,
                        similarity_top_k=self.top_k,
                        #optimizer=SentenceEmbeddingOptimizer(threshold_cutoff=0.1),
                        #response_mode="compact",
                        # mode=self.query_mode,
                        # query_configs = self.query_configs,
                        # available mode : default, retrieve, embedding,
                        #                  summarize, simple, rake, recursive
                        )
            pl(response)
            pl(f"\n##### Sources:\n{response.get_formatted_sources()}\n#####\n\n")
            log.info(f"\nAnswer: {response}")


# handle logging
local_dir = "/".join(__file__.split("/")[:-1])
Path(f"{local_dir}/logs.txt").touch(exist_ok=True)
logging.basicConfig(filename=f"{local_dir}/logs.txt",
                    filemode='a',
                    format=f"{time.asctime()}: %(message)s",
                    level=logging.INFO,
                    force=True,
                    )
log = logging.getLogger()


if __name__ == "__main__":
    def get_args(**kwargs: Dict) -> Dict:
        "used to get arguments from fire.Fire"
        log.info("\n\n\n\n\n")
        log.info("New session. Arguments :")
        pl(kwargs)
        return kwargs
    args = fire.Fire(get_args)

    if "help" in args:
        pl(OmniQA.__doc__)
        pl(OmniQA.__init__.__doc__)
        raise SystemExit()

    omniqa = OmniQA(**args)

    pl("Openning console.")
    while True:
        try:
            code.interact(local=locals())
        except EOFError:
            raise SystemExit()
        except KeyboardInterrupt:
            raise SystemExit()
