PROMPT_DICT = { # alpaca
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "Please do not reply with input.\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT_A1 = { # alpaca
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n"
            "Please do not reply with input.\n"
            "### Instruction:\n{instruction} Answer in a word.\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction} Answer in a word.\n\n### Response:"
        ),
    }


PROMPT_FLAN = {
    "prompt_input": (
        "Answer the following question. {instruction}: "
        #"Please do not reply with input."
        "'{input}'"
    ),
    "prompt_no_input": (       
        "Answer the following question. {instruction}?"
        #"Please do not reply with input."
        #"Can you {instruction}? \n"
    )
}

PROMPT_FLAN_A1 = {
        "prompt_input": (
            "Answer the following question. {instruction}"
            #"Please do not reply with input."
            "'{input}'"
        ),
        "prompt_no_input": (       
            "Answer the following question. {instruction}"
            #"Please do not reply with input."
            #"Can you {instruction}? \n"
        )
    }


PROMPT_BLOOM = {
    "prompt_input": (
        "if the word is {input}.\n"
        "Please answer the following question."
        "Can you {instruction}? \n"
    ),
    "prompt_no_input": (       
        "Please answer the following question."
        "Can you {instruction}? \n"
    )
}

#7,8
PROMPT_BLOOM_A1 = {
    "prompt_input": (
        "Please answer the following question."
        "{instruction}\n"
        "{input}.\n"
    ),
    "prompt_no_input": (  
        "Please answer the following question."
        "{instruction}\n"
    )
}