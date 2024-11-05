import re
import copy
from treelib import Tree
from collections import defaultdict


def read_amr_from_file(from_filename: str):
    from_file = open(from_filename, "r", encoding='utf-8')
    sentences = from_file.readlines()

    amr_structure = []
    amr_list = []

    len_sentences = len(sentences) - 1
    for i, sent in enumerate(sentences):
        split_colon = sent.split(":")
        if split_colon[0].isdigit():
            idx = split_colon[0]
            split_tok = sent.split("::tok ")
            target = split_tok[1][:-1]
        elif sent == "\n":
            amr_structure.append((idx, target, amr_list))
            amr_list = []
        elif i == len_sentences:
            amr_list.append(sent[:-1])
            amr_structure.append((idx, target, amr_list))
        else:
            amr_list.append(sent[:-1])
    return amr_structure


def simplify_amr(amr_structure):
    for item in amr_structure:
        idx, target, amr = item
        # remove leading spaces
        simple_amr = [text.lstrip() for text in amr]

        # remove numbers after tilde
        simple_amr = [re.sub(r'~\d+', '', text) for text in simple_amr]

        # remove number after PropBank framesets (e.g. want-01, hey-02)
        simple_amr = [re.sub(r'-\d+', '', text) for text in simple_amr]

        # remove double quotes
        simple_amr = [text.replace('"', '') for text in simple_amr]

        # remove short names and replace with full names where applicable
        names_dict = {}
        for i, line in enumerate(simple_amr):
            if i == 0 or line[0] == ':':
                if '(' in line:
                    prefix, remainder = line.split('(',
                                                   1)  # split the string at the first occurrence of '(' to separate the prefix
                    if remainder != '' and '/' in remainder:
                        key, value = remainder.split(
                            ' / ')  # split the remainder at ' / ' to separate the key and value
                        simple_amr[
                            i] = f'{prefix}({value}'  # construct the modified string by combining the prefix and the value within the parenthesis
                        names_dict[key] = value
                else:
                    prefix, key = line.split(" ", 1)
                    full_key = key
                    key = key.rstrip(")")
                    ending = ")" * (len(full_key) - len(key))
                    if key in names_dict.keys():
                        full_word = names_dict[key]
                        simple_amr[i] = prefix + ' ' + full_word + ending

        # collapse :op lines
        for i, line in enumerate(simple_amr):
            if line[:4] == ':op1':
                k = i + 1
                while k < len(simple_amr):
                    if simple_amr[k][:3] == ':op' and simple_amr[i][-1] != ')':
                        simple_amr[i] += simple_amr[k][4:]
                        simple_amr[k] = ""
                        k += 1
                    else:
                        break

        # collapse name lines
        for i, line in enumerate(simple_amr):
            if line[:5] == ':name' and line[-4:] == 'name':
                if idx == '28456':
                    bla = 2
                simple_amr[i] = ""
                simple_amr[i + 1] = simple_amr[i + 1][:-1]
                if simple_amr[i + 1][:4] == ':op1':
                    simple_amr[i + 1] = simple_amr[i + 1][5:]

        # remove empty lines
        simple_amr = [line for line in simple_amr if line]

        # remove parenthesis for singe words
        for i, line in enumerate(simple_amr):
            split_list = line.split()
            if len(split_list) > 1 and split_list[1][0] == '(' and split_list[1][-1:] == ')':
                simple_amr[i] = split_list[0] + ' ' + split_list[1][1:-1]

        # join the items
        simple_amr = " ".join(simple_amr)

        print(idx, simple_amr)


# calculate the number of failed parsings (contain "rel" tag)
def find_failed_parsing(amr_structure):
    with open("data/failed_parsings.txt", "w", encoding='utf-8') as file:
        failed_count = 0
        for i, item in enumerate(amr_structure):
            idx, target, amr = item
            for line in amr:
                if ':rel' in line:
                    failed_count += 1
                    full_parsing = '\n'.join(amr)
                    file.write("idx %s, %d out of %d:\n%s\n%s\n\n" % (idx, failed_count, i + 1, target, full_parsing))
                    break


def build_amr_tree(amr_structure_item, remove_word_order=True):
    idx, target, amr = amr_structure_item
    # remove leading spaces
    simple_amr = [text.lstrip() for text in amr]
    # remove numbers after tilde
    if remove_word_order:
        simple_amr = [re.sub(r'~\d+', '', text) for text in simple_amr]
    # remove number after PropBank framesets (e.g. want-01, hey-02) BUT leave numbers after spaces (e.g. temperature -1)
    simple_amr = [re.sub(r'(?<=\S)-\d+', '', text) for text in simple_amr]

    def slash_split(line):
        if "/" in line:
            n1, n2 = line.split(" / ", 1)
            names_dict[n1] = n2
            return n1, n2
        else:
            return line, 0

    def single_token_split(line):
        # to catch corner cases where the token after operator is not a short name
        # like ":polarity -", ":mod B", ":mode expressive"
        if (line[0] == '"' or line[0].isdigit() or line[0] == '-' or line.isupper() or len(line) > 2
                or line in {'he', 'it', '.', '+'}):
            return 0, line
        else:
            return line, 0

    # create tree
    tree = Tree()
    prev_node_id = 0
    cur_id = 0
    names_dict = {}  # dictionary for short/long names
    for i, line in enumerate(simple_amr):
        if i == 0:
            line = line.lstrip("(")
            d1, d2 = slash_split(line)
            tree.create_node(tag="root", identifier=0, data=(d1, d2))
            prev_node_id = tree[0].identifier
        else:
            tag, remainder = line.split(" ", 1)
            if remainder[0] == "(" and remainder[-1:] == ")":
                remainder = remainder[1:len(remainder) - 1]
                new_remainder = remainder.rstrip(")")
                step_up = len(remainder) - len(new_remainder)
                d1, d2 = slash_split(new_remainder)

                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
                for _ in range(step_up):
                    prev_node_id = tree.get_node(prev_node_id).predecessor(tree.identifier)
            elif remainder[0] == "(":
                remainder = remainder[1:]
                d1, d2 = slash_split(remainder)
                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
                prev_node_id = tree[cur_id].identifier
            elif remainder[-1:] != ")":
                d1, d2 = single_token_split(remainder)
                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
            else:
                new_remainder = remainder.rstrip(")")
                d1, d2 = single_token_split(new_remainder)
                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
                step_up = len(remainder) - len(new_remainder)
                for _ in range(step_up):
                    prev_node_id = tree.get_node(prev_node_id).predecessor(tree.identifier)
    return tree, names_dict


def collapse_leaves(tree, names_dict):
    temp_tree = copy.deepcopy(tree)
    for node in temp_tree.all_nodes_itr():
        if node.is_leaf() and (tree.depth(node.identifier) > 2 or node.data[0] == 0 or node.tag == ":name"):
            parent_id = node.predecessor(tree.identifier)
            parent = tree[parent_id]

            # update the long name
            if node.data[0] != 0:
                node.data = (node.data[0], names_dict[node.data[0]])

            data = '' if parent.data[1] == 'name' else str(parent.data[1]) + ' '

            parent.data = (parent.data[0], data + node.data[1])
            tree.remove_node(node.identifier)
            # update the dict with short/long names
            names_dict[parent.data[0]] = parent.data[1]
    return tree, names_dict


def clean_names_dict(names_dict):
    for key in names_dict:
        name = names_dict[key]

        words = name.split()[::-1]
        words_copy = copy.deepcopy(words)

        # remove one word before words in quotes or words before numbers
        # example: govern city "Birmingham" -> govern "Birmingham"
        # example: date-entity 1981 -> 1981
        if len(words) > 1:
            for i, word in enumerate(words):
                if i + 2 < len(words) and word[0] == '"' and words[i + 1][0] != '"' and words[i + 2][0] == '"':
                    words_copy[i + 1] = ','
                elif i + 1 < len(words) and word[0] == '"' and words[i + 1][0] != '"':
                    words_copy[i + 1] = ''
                elif i + 1 < len(words) and word.isdigit() and words[i + 1][0] != '"' and not words[i + 1][0].isdigit():
                    words_copy[i + 1] = ''
        words = words_copy[::-1]

        # remove empty lines
        words = [word for word in words if word]
        # make one line
        words = " ".join(words)
        # remove double quotes
        words_str = words.replace('"', '')

        names_dict[key] = words_str
    return names_dict


def build_one_line_tree(tree):
    one_line = ''
    prev_node_id = 0
    ending_brackets_counter = 0

    # helper function to append node data based on whether it has children or not (embedding)
    def append_node(node, has_children):
        nonlocal one_line, ending_brackets_counter
        tag = re.sub(r'\d+', '', node.tag) if node.tag[:4] == ':ARG' else node.tag
        if has_children:
            one_line += f" {tag} ({node.data[1]}"
            ending_brackets_counter += 1
        else:
            one_line += f" {tag} {node.data[1]}"

    for node in tree.all_nodes_itr():
        if node.tag == 'root':
            one_line = node.data[1]
        else:
            # get node's parent and prev node's parent
            parent_id = node.predecessor(tree.identifier)
            prev_parent_id = tree[prev_node_id].predecessor(tree.identifier)
            has_children = len(tree.children(node.identifier)) > 0

            # if node is under the prev one or on the same level
            if parent_id == prev_node_id or parent_id == prev_parent_id:
                append_node(node, has_children)
            else:
                # backtrack adding closing parentheses for prev nodes
                prev_node_id = tree[prev_node_id].predecessor(tree.identifier)
                while parent_id != prev_node_id:
                    prev_node_id = tree[prev_node_id].predecessor(tree.identifier)
                    one_line += ')'
                    ending_brackets_counter = 0

                append_node(node, has_children)

        prev_node_id = node.identifier

    one_line = one_line + ")" * ending_brackets_counter
    return one_line


def print_tree(tree):
    tree_bytes = tree.show(stdout=False, sorting=False)
    print(tree_bytes.encode('utf-8').decode('utf-8'))


def print_tags(tree):
    for node in tree.all_nodes_itr():
        print(node.tag, node.data)
    print("\n")


# replace phrases in simplified tree with the phrase from surface sentence
# ex: simplified node: :ARG-of 'have-org-role chairman Silvio Berlusconi'
# search for max and min word orders in an initial tree for :ARG-of node (from 8 to 11)
# and replace: :ARG-of 'managed by Sinisa Mihajlovic'
def replace_with_surface_phrase(node, init_tree, target_sent):
    subtree = init_tree.subtree(node.identifier)
    if '~' not in str(init_tree[node.identifier].data[1]):
        return False
    max_ord = min_ord = int(re.search(r'~(\d+)', init_tree[node.identifier].data[1]).group(1))
    for n in subtree.all_nodes_itr():
        if '~' in str(n.data[1]):
            word_order = int(re.search(r'~(\d+)', n.data[1]).group(1))
            max_ord = max(word_order, max_ord)
            min_ord = min(word_order, min_ord)

    max_min_diff = max_ord - min_ord
    if max_min_diff < 8:
        surface_list = target_sent.split()
        surface_phrase = " ".join(surface_list[min_ord:max_ord + 1])
        node.data = (node.data[0], surface_phrase)
        return True
    else:
        return False


def simplify_tree_test(amr_structure):
    for item in amr_structure:
        init_tree, _ = build_amr_tree(item, remove_word_order=False)
        tree, names_dict = build_amr_tree(item, remove_word_order=True)

        idx, target, amr = item

        print('\n', "target sentence:")
        print(target, '\n')

        print("AMR parsing:")
        for a in amr:
            print(a)

        print("initial tree:")
        print_tree(tree)

        # add long names where possible
        for node in tree.all_nodes_itr():
            if node.data[1] == 0:
                node.data = (node.data[0], names_dict[node.data[0]])

        print("initial tags:")
        print_tags(tree)

        for _ in range(3):
            tree, names_dict = collapse_leaves(tree, names_dict)

        print("collapsed tags:")
        print_tags(tree)

        # simplify the long names
        names_dict = clean_names_dict(names_dict)
        replaced = set()
        for node in tree.all_nodes_itr():
            key = node.data[0]
            value = names_dict[node.data[0]]

            # update node data by default
            node.data = (key, value)

            if len(value.split()) >= 4 and key not in replaced:
                if replace_with_surface_phrase(node, init_tree, target):
                    names_dict[key] = node.data[1]
                    replaced.add(key)

        print("simplified tags and tree:")
        print_tags(tree)
        print_tree(tree)

        simplified_amr = build_one_line_tree(tree)
        print("simplified one-line ARM:")
        print(simplified_amr)


def simplify_tree_save_to_file(amr_structure, filename):
    simple_tree_list = []
    with open(filename, "w", encoding='utf-8') as file:
        for item in amr_structure:
            init_tree, _ = build_amr_tree(item, remove_word_order=False)
            tree, names_dict = build_amr_tree(item, remove_word_order=True)
            idx, target, amr = item

            # add long names where possible
            for node in tree.all_nodes_itr():
                if node.data[1] == 0:
                    node.data = (node.data[0], names_dict[node.data[0]])

            for _ in range(3):
                tree, names_dict = collapse_leaves(tree, names_dict)

            # simplify the long names
            names_dict = clean_names_dict(names_dict)
            replaced = set()
            for node in tree.all_nodes_itr():
                key = node.data[0]
                value = names_dict[node.data[0]]

                # update node data by default
                node.data = (key, value)

                if len(value.split()) >= 4 and key not in replaced:
                    if replace_with_surface_phrase(node, init_tree, target):
                        names_dict[key] = node.data[1]
                        replaced.add(key)

            simplified_tree = build_one_line_tree(tree)
            simple_tree_list.append(simplified_tree)

            file.write("%s\n" % idx)
            file.write("%s\n" % target)
            file.write("%s\n" % simplified_tree)

    return simple_tree_list


# collapse the leaves and simplify long names
def simplify_tree(amr_structure):
    simple_tree_list = []
    for item in amr_structure:
        init_tree, _ = build_amr_tree(item, remove_word_order=False)
        tree, names_dict = build_amr_tree(item, remove_word_order=True)
        idx, target, amr = item

        # add long names where possible
        for node in tree.all_nodes_itr():
            if node.data[1] == 0:
                node.data = (node.data[0], names_dict[node.data[0]])

        for _ in range(3):
            tree, names_dict = collapse_leaves(tree, names_dict)

        # simplify the long names
        names_dict = clean_names_dict(names_dict)
        replaced = set()
        for node in tree.all_nodes_itr():
            key = node.data[0]
            value = names_dict[node.data[0]]

            # update node data by default
            node.data = (key, value)

            if len(value.split()) >= 4 and key not in replaced:
                if replace_with_surface_phrase(node, init_tree, target):
                    names_dict[key] = node.data[1]
                    replaced.add(key)

        simplified_tree = build_one_line_tree(tree)
        simple_tree_list.append((idx, simplified_tree))
    return simple_tree_list


def save_to_file(data: list, filename: str):
    with open(filename, "w", encoding='utf-8') as file:
        cur_idx = '1'
        first_line = True

        for idx, item in data:
            wrapped_item = f'[{item}]'

            if first_line:
                file.write(wrapped_item)
                first_line = False
            elif idx == cur_idx:
                file.write(f' {wrapped_item}')
            else:
                file.write(f'\n{wrapped_item}')
            cur_idx = idx


if __name__ == '__main__':
    # amr_structure = read_amr_from_file("data/amr_parser_results_example.txt")
    amr_structure = read_amr_from_file("data/amr_parser_results_test.txt")

    # find_failed_parsing(amr_structure)

    # simplify_tree_test(amr_structure)
    simplified_amr = simplify_tree(amr_structure)
    # simplified_amr = simplify_tree_save_to_file(amr_structure, "data/amr_parser_simplified_w_targets_train_4.txt")
    save_to_file(simplified_amr, "data/amr_parser_simplified_test.txt")


