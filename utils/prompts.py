from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


class PromptClass:

    french_rules = """Voilà les règles que vous devez imperativement suivre :
    1. Si le contexte est pertinent, commencez par "OK" puis répondez en citant le contexte fournis.
    2. Si le contexte est partiellement pertinent, commencez par "OK" puis repondez en citant le contexte puis "NOSOURCE" suivi de vos propres connaissances.
    3. Si le contexte n'est pas pertinent, commencez par "NOSOURCE" puis répondez avec vos connaissances les plus solides.
    4. Utilisez un ton impartial, académique et en français.
    5. Soyez concis, lisible, en aérant votre réponse en allant a la ligne.
    6. Le cas échéant, formatez votre réponse à l'aide de puces Markdown."""

    english_rules = """Here are the rules you absolutely have to follow:
    1. If the context is relevant, begin your answer by "SOURCED" then answer using quotes from the context.
    2. If the context is partially relevant, begin by "SOURCED" then answer using quotes from the context then "NOSOURCE" followed by your own knowledge.
    3. If the context is NOT relevant, begin your answer by "NOSOURCE" then answer as truthfully as you can.
    4. Use an unbiased, scholarly tone and in english.
    5. Be concise yet easily readable by using newlines.
    6. When applicable, format your answer using markdown bullet points."""

    french_qa = """Nous allons vous fournir un contexte d'information, des règles à suivre ainsi qu'une question.
    """ + french_rules + """
    La question est la suivante : '{query_str}'
    Voici le contexte :
    --------------------
    {context_str}
    --------------------
    """

    french_qa_default_refine = """Question initiale : '{query_str}'
    """ + french_rules + """
    Votre réponse : '{existing_answer}'
    Vous avez la possibilité de continuer votre réponse grace au contexte suivant. Ne le faites que si le contexte est pertinent. Respectez les règles initiales, commencez impérativement votre réponse directement par votre réponse initiale.
    Voici les nouveau contexte:
    --------------------
    {context_msg}
    --------------------
    """

    french_qa_chat_refine = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    """Question initiale : {query_str}
                    """ + french_rules + """
                    Votre réponse initiale : '{existing_answer}'
                    Nous allons vous donner du contexte additionnel pour vous permettre de completer votre réponse. En aucun cas pouvez vous modifier votre réponse initiale, vous pouvez uniquement rajouter du texte à sa suite. Ne le faite que si le nouveau contexte s'y prete, sinon répondez uniquement votre réponse initiale.
                    Avant de continuer, pouvez vous confirmer que vous être prêts ?"""),
                AIMessagePromptTemplate.from_template(
                    "Oui. Je suis prêt à lire le nouveau contexte."),
                HumanMessagePromptTemplate.from_template(
                    """Voici le nouveau contexte :
                    --------------------
                    {context_msg}
                    --------------------
                    N'oubliez pas de commencer votre réponse par la réponse initiale, et de ne continuer votre réponse que si nécéssaire.
                    """)
                ]
            )


    english_qa = """We will give you some context information, a set of rules and then a question.
    """ + english_rules + """
    Here's the question: '{query_str}'
    Here's the context:
    --------------------
    {context_str}
    --------------------
    """

    english_qa_default_refine = """Initial question: '{query_str}'
    """ + english_rules + """
    Your initial answer: '{existing_answer}'
    You have the occasion to continue your initial answer using the the following context. Only do it if the next context is relevant to the question. Respect the initial rules. It is very important that you answer begins direclty with your initial answer and to only add new content at the end (if relevant).
    Here's the new context:
    --------------------
    {context_msg}
    --------------------
    """

    english_qa_chat_refine = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    """Initial question : {query_str}
                    """ + english_rules + """
                    Your initial answer: '{existing_answer}'
                    We will give you additional context to allow you to continue your answer. In no circumstances are you allowed to modify your initial answer, you can only add new text at the end. Only add new text if the new context is relevant. If it's not relevant, only replu your initial answer.
                    Before continuing, can you confirm you're up for the task?"""),
                AIMessagePromptTemplate.from_template(
                    "Yes. I am ready to read the new context."),
                HumanMessagePromptTemplate.from_template(
                    """Here's the new context:
                    --------------------
                    {context_msg}
                    --------------------
                    Don't forget to begin your answer using your initial answer, and to only add new content after it if needed.
                    """
                    )
                ]
            )
