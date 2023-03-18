from pathlib import Path
import time
from llama_index import download_loader, SimpleDirectoryReader, Document


class ParserClass:
    def __init__(self, printer):
        self.pl = printer
    
    def dispatch(self, path, extra_docid, import_type=None):
        """
        detects the appropriate file type and handles it accordingly
        """
        try:
            if str(path).lower().endswith(".pdf"):
                return self.handle_pdf(path, extra_docid)
            elif str(path).lower().endswith(".md"):
                return self.handle_markdown(path, extra_docid)
            elif str(path).lower().endswith(".txt"):
                return self.handle_markdown(path, extra_docid)
            else:
                self.pl("File with extension other than pdf or md "
                        f"found: '{path}'")
                self.pl("Trying to parse it.")
                return self.handle_generic(path=path, docid=extra_docid, import_type=import_type)
        except Exception as err:
            self.pl(f"Exception when adding '{path}': '{err}'. Skipping this file.")
            return None

    def handle_pdf(self, path, extra_docid):
        # load from llama hub:
        # pdf_reader = download_loader("CJKPDFReader")
        pdf_reader = download_loader("PDFReader")
        # ^- issue with too many weird characters?
        loader = pdf_reader()

        docid = path.split("/")[-1] + f"_{int(time.time())}"
        if extra_docid:
            docid = f"{extra_docid}_{docid}"
        new_text = loader.load_data(file=path)
        assert len(new_text) == 1, "Invalid length of new_text"
        new_doc = Document(new_text[0].text, doc_id=docid)
        if isinstance(new_doc.text, list):
            # for some reason sometimes its a list and causing issues
            new_doc.text = new_doc.text[0]
        self.pl(f"    * Loaded as pdf '{path}'\n")
        return new_doc

    def handle_markdown(self, path, extra_docid):
        docid = path.split("/")[-1] + f"_{int(time.time())}"
        if extra_docid:
            docid = f"{extra_docid}_{docid}"
        new_doc = Document(Path(path).read_text(), doc_id=docid)
        return new_doc

    def handle_generic(self, path, docid, import_type):
        # alternative without llama hub:
        new_doc = SimpleDirectoryReader(input_files=[path],
                                        doc_id=[docid]).load_data()
        assert isinstance(new_doc.text, str), "Doc.text is not string."
        assert len(new_doc.text)>10, "Less than 10 characters found."
        return new_doc
