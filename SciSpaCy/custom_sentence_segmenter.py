def combined_rule_splitter(doc):
    # keep track of the two previous tokens
    prev_tokens = [None, None]

    # keep stacks for determining when we are inside parenthesis or brackets
    open_parens = []
    open_brackets = []
    for token in doc:
        # handling special quote symbols
        if token.text == '“' or token.text == '”':
            if prev_tokens[-1] and prev_tokens[-1].text != ".":
                doc[token.i].is_sent_start = False

        if token.text[0].isdigit():
            # handling an abbrevation followed by a number
            abbreviations = ["sec.", 
                             "secs.", 
                             "Sec.", 
                             "Secs.", 
                             "fig.", 
                             "figs.", 
                             "Fig.", 
                             "Figs.", 
                             "eq.", 
                             "eqs.", 
                             "Eq.", 
                             "Eqs.", 
                             "no.", 
                             "nos.", 
                             "No.", 
                             "Nos."]
            if prev_tokens[-1] and prev_tokens[-1].text in abbreviations:
                doc[token.i].is_sent_start = False
            
            # handle a bracket followed by a number
            if prev_tokens[-1] and prev_tokens[-1].text == '[':
                doc[token.i].is_sent_start = False
            

        # sentences can only start with ( if there is a complete sentence within the parens
        # here a . is serving as a proxy for being a complete sentence
        # This isn't quite correct, as a sentence may start with (Bottom left) and this prevents that 
        # from being segmented correctly
        if token.text == "(":
            open_parens.append(token)
        if token.text == ")" and open_parens != []:
            last_open_paren = open_parens.pop()
            if prev_tokens[-1] and prev_tokens[-1].text != ".":
                # allow things like (A) to start a sentence
                if not (last_open_paren.i == (token.i-2) and len(prev_tokens[-1].text) == 1):
                    doc[last_open_paren.i].is_sent_start = False
        
        # same logic as above but for brackets instead of parens
        if token.text == "[":
            open_brackets.append(token)
        if token.text == "]" and open_brackets != []:
            last_open_bracket = open_brackets.pop()
            if prev_tokens[-1] and prev_tokens[-1].text != ".":
                # allow things like [A] to start a sentence
                if not (last_open_bracket.i == (token.i-2) and len(prev_tokens[-1].text) == 1):
                    doc[last_open_bracket.i].is_sent_start = False
        
        # handling the case of a capital letter after a ) unless that was preceeded by a .
        first_char = token.text[0]
        if first_char.isupper():
            if prev_tokens[-1] and prev_tokens[-1].text == ')':
                if prev_tokens[-2] and prev_tokens[-2].text != ".":
                    doc[token.i].is_sent_start = False

        # sentences cannot start with .
        if token.text == ".":
            doc[token.i].is_sent_start = False

        # attempt to handle double and quadruple new lines around section headers by making them their own sentences
        if prev_tokens[-1] and (prev_tokens[-1].text == "\n\n\n\n" or prev_tokens[-1].text == "\n\n"):
            doc[token.i].is_sent_start = True
            doc[token.i-1].is_sent_start = True

        # update the saved previous tokens
        prev_tokens = prev_tokens[1:] + [token]

    # any unmatched parens should not be the start of a sentence
    for open_paren in open_parens:
        doc[open_paren.i].is_sent_start = False

    # any unmatched brackets should not be the start of a sentence
    for open_bracket in open_brackets:
        doc[open_bracket.i].is_sent_start = False

    return doc