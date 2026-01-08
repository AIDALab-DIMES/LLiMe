import re
import nltk
import spacy
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy English model for NLP tasks
nlp = spacy.load("en_core_web_sm")
# Download all NLTK data (quietly to suppress output)
nltk.download('all', quiet=True)


def parse_LLM_output(llm_output, path=None):
    """
    Parse LLM-generated counterfactual explanations to extract two key components:
    - C: The counterfactual sentence (alternative sentence with flipped prediction)
    - O: List of edit operations to transform original to counterfactual

    The function handles multiple output formats from different LLM responses, particularly, we considered GPT and
    DeepSeek family Models; it may be necessary to redefine the parsing for others LLMs.

    Args:
        llm_output: Raw text output from the language model
        path: Optional directory path to save parsed results to files

    Returns:
        tuple: (c, o)
            - c: Counterfactual sentence string
            - o: Edit operations string (typically a list representation)

    Raises:
        Exception: If counterfactual or operation list cannot be found in output


    """
    llm_output += "\n"  # Add newline for consistent parsing
    llm_output_lower = llm_output.lower()

    # Try multiple patterns to find the counterfactual sentence "C"
    # Handles variations: **C** =, "C" =, 'C' =, **"C"** =, C =, "C":
    if "**c** = " in llm_output_lower:
        id = llm_output_lower.index("**c** = ") + len("**c** = ")
        c = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
        c = c[c.index("\"") + 1: c.rindex("\"")].replace("\\", "")
    elif "\"c\" = " in llm_output_lower:
        id = llm_output_lower.index("\"c\" = ") + len("\"c\" = ")
        c = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
        c = c[c.index("\"") + 1: c.rindex("\"")].replace("\\", "")
    elif "\'c\' = " in llm_output_lower:
        id = llm_output_lower.index("\'c\' = ") + len("\'c\' = ")
        c = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
        c = c[c.index("\"") + 1: c.rindex("\"")].replace("\\", "")
    elif "**\"c\"** = " in llm_output_lower:
        id = llm_output_lower.index("**\"c\"** = ") + len("**\"c\"** = ")
        c = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
        c = c[c.index("\"") + 1: c.rindex("\"")].replace("\\", "")
    elif "c = " in llm_output_lower:
        id = llm_output_lower.index("c = ") + len("c = ")
        c = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
        c = c[c.index("\"") + 1: c.rindex("\"")].replace("\\", "")
    elif "\"c\": " in llm_output_lower:
        id = llm_output_lower.index("\"c\": ") + len("\"c\": ")
        c = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
        c = c[c.index("\"") + 1: c.rindex("\"")].replace("\\", "")
    else:
        raise Exception("COUNTERFACTUAL NOT FOUND:\n", llm_output)

    # Try multiple patterns to find the operations list "O"
    # Handles variations: **O** =, "O" =, 'O' =, **"O"** =, O =, "O":
    if "**o** = " in llm_output_lower:
        id = llm_output_lower.index("**o** = ") + len("**o** = ")
        o = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
    elif "\"o\" = " in llm_output_lower:
        id = llm_output_lower.index("\"o\" = ") + len("\"o\" = ")
        o = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
    elif "\'o\' = " in llm_output_lower:
        id = llm_output_lower.index("\'o\' = ") + len("\'o\' = ")
        o = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
    elif "**\"o\"** = " in llm_output_lower:
        id = llm_output_lower.index("**\"o\"** = ") + len("**\"o\"** = ")
        o = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
    elif "o = " in llm_output_lower:
        id = llm_output_lower.index("o = ") + len("o = ")
        o = llm_output[id:llm_output_lower.index("\n", id, len(llm_output))]
    elif "\"o\": " in llm_output_lower:
        id = llm_output_lower.index("\"o\": ") + len("\"o\": ")
        # Extract until closing bracket, remove newlines and clean formatting
        o = llm_output[id:1 + llm_output_lower.index("]", id, len(llm_output))].replace('\n', '')
        o = o.replace("\"SUB", "SUB").replace("\"REM", "REM").replace("\"INS", "INS").replace(')"', ')')
        o = o.replace("\'SUB", "SUB").replace("\'REM", "REM").replace("\'INS", "INS").replace(")\'", ")")
    else:
        raise Exception("LIST OF OPERATION NOT FOUND:\n", llm_output)

    # Optionally save results to files
    if path != None:
        print(c, file=open(f"{path}/counterfactual.txt", "w"))
        print(o, file=open(f"{path}/edit_operations.txt", "w"))

    return c, o


def create_llime_explanation(sentence, edit_operations, path=None):
    """
    Convert edit operations string into a structured DataFrame explanation.

    Parses a list of edit operations (SUB, REM, INS) and organizes them into
    a tabular format showing what changes transform the original sentence
    into the counterfactual.

    Operations:
    - SUB(word1, word2): Substitute word1 with word2
    - REM(word): Remove word
    - INS(word): Insert word

    Args:
        sentence: Original sentence being explained
        edit_operations: String representation of list of operations
        path: Optional directory path to save explanation CSV

    Returns:
        pandas.DataFrame with columns:
            - OPERATION: Type of edit (SUB, REM, or INS)
            - OUT_WORD: Word being removed/replaced (or " " for INS)
            - IN_WORD: Word being added/inserted

    Raises:
        Exception: If operation parsing fails
    """
    # Parse the operations list string into individual operations
    # Format: [SUB(w1,w2), REM(w), INS(w)] -> ["SUB(w1,w2)", "REM(w)", "INS(w)"]
    edit_operations = edit_operations[1:edit_operations.index(']')].replace('), ', ')_').replace('),', ')_').split('_')
    df = pd.DataFrame(index=range(len(edit_operations)), columns=["OPERATION", "OUT_WORD", "IN_WORD"])

    for i, edit_operation in enumerate(edit_operations):
        try:
            # Clean the operation string
            eo = edit_operation.replace("\"", "").replace(" ", "").replace("'", "")
            op = eo[:3]  # Extract operation type (first 3 chars)

            if op == "REM" or op == "INS":
                # REM and INS have single word: REM(word) or INS(word)
                df.iloc[i] = [op, " ", eo[4:-1]]
            else:
                # SUB has two words: SUB(word1, word2)
                w1, w2 = eo[4:-1].split(",")
                # Determine which word is in original sentence (OUT_WORD)
                if w1 in sentence:
                    df.iloc[i] = [op, w2, w1]  # w1 is OUT (original), w2 is IN (new)
                else:
                    df.iloc[i] = [op, w1, w2]  # w2 is OUT (original), w1 is IN (new)
        except:
            print(edit_operation)
            raise Exception("ERROR IN PARSING OPERATION!")

    # Optionally save to CSV
    if path != None:
        df.dropna().to_csv(f"{path}/llime.csv")

    return df


def verify(parola, frase):
    """
    Check if a word exists as a complete word in a sentence (word boundary matching).

    Uses regex word boundaries to ensure exact word matches, preventing partial
    matches (e.g., "cat" won't match "category").

    Args:
        parola: Word to search for (Italian: "word")
        frase: Sentence to search in (Italian: "sentence")

    Returns:
        bool: True if word found as complete word, False otherwise
    """
    # Build pattern with word boundaries (\b) to match complete words only
    # re.escape handles special regex characters in the word
    pattern = r'\b' + re.escape(parola) + r'\b'
    # Search for the word in the sentence
    return re.search(pattern, frase) is not None


def check_explanation(sentence, counterfactual, explanation):
    """
    Validate that edit operations correctly transform sentence to counterfactual.

    Checks each operation in the explanation DataFrame to verify it correctly
    describes a change between the original sentence and counterfactual.
    Prints warnings for operations that don't match expected transformations.

    Validation rules:
    - INS: Word should be in counterfactual but NOT in original
    - REM: Word should be in original but NOT in counterfactual
    - SUB: One word should be in original, the other in counterfactual

    Args:
        sentence: Original sentence
        counterfactual: Transformed sentence
        explanation: DataFrame of edit operations (from create_llime_explanation)

    Returns:
        None (prints warnings for missing/incorrect operations)
    """
    sentence = sentence.lower()
    counterfactual = counterfactual.lower()

    for i in range(explanation.shape[0]):
        row = explanation.iloc[i]
        op = row.iloc[0]  # Operation type
        w1 = row.iloc[1].lower().replace('#', '')  # OUT_WORD
        w2 = row.iloc[2].lower().replace('#', '')  # IN_WORD

        if op == "INS":
            # Check: w2 should be in counterfactual and NOT in sentence
            if not (verify(w2, counterfactual) and not verify(w2, sentence)):
                print("MISSING OPERATION:")
                print(sentence)
                print(counterfactual)
                print(op, w1, w2)

        elif op == "REM":
            # Check: w2 should be in sentence and NOT in counterfactual
            if not (not verify(w2, counterfactual) and verify(w2, sentence)):
                print("MISSING OPERATION:")
                print(sentence)
                print(counterfactual)
                print(op, w1, w2)

        else:  # SUB
            # Check: Either (w2 in counterfactual AND w1 in sentence) OR vice versa
            if not ((verify(w2, counterfactual) and verify(w1, sentence)) or (
                    verify(w1, counterfactual) and verify(w2, sentence))):
                print("MISSING OPERATION:")
                print(sentence)
                print(counterfactual)
                print(op, w1, w2)

    return


def split_sentence(sentence):
    """
    Tokenize a sentence into words while preserving named entities (PERSON names).

    Uses spaCy to:
    1. Tokenize the sentence
    2. Merge multi-word PERSON entities into single tokens (e.g., "John Smith")
    3. Filter to only alphabetic tokens

    This is important for explanation methods to treat names as single features
    rather than splitting them into separate tokens.

    Args:
        sentence: Input sentence string

    Returns:
        list: Filtered tokens containing at least one alphabetic character
    """
    # Process sentence with spaCy NLP pipeline
    doc = nlp(sentence)

    # Merge tokens that are part of a PERSON named entity
    # Example: "John" + "Smith" -> "John Smith" (single token)
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                retokenizer.merge(ent)

    # Extract token texts
    tokens = [token.text for token in doc]

    # Filter to keep only tokens containing at least one letter
    # Removes pure punctuation, numbers, symbols
    to_ret = []
    for token in tokens:
        if bool(re.search(r'[a-zA-Z]', token)):
            to_ret.append(token)

    return to_ret