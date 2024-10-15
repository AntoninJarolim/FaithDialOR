import json
import os
import shutil
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

faith_dial_datafiles = ['test.json', 'train.json', 'valid.json',
                        'test_random_split.json', 'valid_random_split.json',
                        'test_topic_split.json', 'valid_topic_split.json']

fd_fixes_mapping = {
    # Typo 'sure' -> 'surf'
    'I absolutely love to surf, just riding on the forward face of a moving wave is so exhilarating. Are there a lot of sharks near you?': 'I absolutely love to sure, just riding on the forward face of a moving wave is so exhilarating. Are there a lot of sharks near you?',
    # Typo 'russian' -> 'rusain'
    'Yes, my uncle was russain and effected by the chernobyl disaster. He had a hard time trying to have a kid because of the radiation cancer he had to deal with.': 'Yes, my uncle was rusain and effected by the chernobyl disaster. He had a hard time trying to have a kid because of the radiation cancer he had to deal with.',
    # removed 'I hope that is correct.'
    "Let's see. If memory serves me correctly,  Hans Heinrich Josef Meyer  was a geographer and geologist from Germany who also climbed mountains and volcanic peaks?  ": 'Let\'s see. If memory serves me correctly,  Hans Heinrich Josef Meyer  was a geographer and geologist from Germany who also climbed mountains and volcanic peaks?  I hope that is correct.'
}


def download_faith_dial():
    for data_name in faith_dial_datafiles:
        hf_hub_download(
            repo_id='McGill-NLP/FaithDial',
            filename=f'data/{data_name}',
            repo_type="dataset",
            local_dir="./FaithDial",
            local_dir_use_symlinks=False  # Ensure direct download, no symlinks
        )


def print_dialogue(filename, max_id):
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            print(json.dumps(data[:max_id], indent=4))


def try_find_knowledge_sent(knowledge_sent, wow_knowledge, diag):
    """
     'original_response' is None for some utterances
     in that case, we try to find the knowledge not baseed on 'original_response'
     but based on the knowledge sentence itself
     this function goes through the dialogue and tries to find the knowledge passage
    :param knowledge_sent: we are trying to find this sentence in the knowledge passages
    :param wow_knowledge: 'original_response' -> knowledge dict
    :param diag: current dialogue we are searching in
    :return: knowledge key if found, None otherwise
    """
    for utt in diag['utterances']:
        if utt["original_response"] is None:
            continue
        knowledge_key = utt["original_response"]
        knowledge_key = fd_fixes_mapping.get(knowledge_key, knowledge_key)

        knowledge = wow_knowledge.get(knowledge_key)
        for passage in knowledge['retrieved_passages']:
            if knowledge_sent in passage:
                return knowledge_key


def find_in_hard_knowledge(knowledge_sentence, wow_hard_knowledge):
    for k, v in wow_hard_knowledge.items():
        if knowledge_sentence in v:
            return True
    return False


def create_ORFaithDial(data_file, wow_knowledge, wow_hard_knowledge):
    not_found = 0
    found_in_other_utterances = 0
    all_utterances = 0
    found_knowledge_sent = 0
    found_normally = 0
    skip_key_error = 0

    faith_dial_data = json.load(data_file)
    for diag in tqdm(faith_dial_data):
        for utt in diag['utterances']:
            all_utterances += 1
            knowledge_key = utt["original_response"]
            knowledge_key = fd_fixes_mapping.get(knowledge_key, knowledge_key)

            if knowledge_key is None:
                knowledge_key = try_find_knowledge_sent(utt["knowledge"].strip(), wow_knowledge, diag)
                if knowledge_key is None:
                    found = find_in_hard_knowledge(utt["knowledge"], wow_hard_knowledge)
                    if found:
                        found_knowledge_sent += 1
                        continue
                else:
                    found_in_other_utterances += 1
            else:
                found_normally += 1

            if knowledge_key is None:
                not_found += 1
                if not_found < 10:
                    print("'{}'".format(utt['knowledge']))
                continue

            try:
                knowledge = wow_knowledge[knowledge_key]
            except KeyError:
                skip_key_error += 1
                continue  # Skip this utterance

            utt['passages'] = knowledge['retrieved_passages']
            utt['topics'] = knowledge['retrieved_topics']
            utt['checked_sentence'] = knowledge['checked_sentence']
            utt['checked_passage'] = knowledge['checked_passage']

            # todo check if utt['knowledge'] is in the retrieved passages

    print(f"Out of {all_utterances} utterances:")
    print(f"\tFound based on 'original_response': {found_normally}")
    print(f"\tFound in other utterances: {found_in_other_utterances}")
    print(f"\tFound sentence in all knowledge: {found_knowledge_sent}")
    print(f"\tNot found anywhere: {not_found}")
    if skip_key_error > 0:
        print(f"\tSkipped because of KeyError: {skip_key_error}")

    sum_found = found_normally + found_in_other_utterances + found_knowledge_sent
    skip_sum = not_found + skip_key_error
    assert all_utterances == sum_found + skip_sum

    return faith_dial_data


def load_wow_knowledge():
    # Replace 'filename.txt' with your actual file name
    wow_knowledge = {}
    wow_hard_knowledge = {}

    with open(f'data/wizard_of_wikipedia/data.json', 'r') as file:
        for line in file:
            data = json.loads(line)
            for diagid, diag in enumerate(data):
                for utt in diag["dialog"]:
                    if utt['speaker'] != "1_Wizard":
                        continue

                    knowledge_key = utt["text"].strip()
                    knowledge_key = knowledge_key.replace("\"", "''")
                    wow_knowledge[knowledge_key] = {
                        'checked_sentence': utt["checked_sentence"],
                        'checked_passage': utt["checked_passage"],
                        'retrieved_passages': [v for topic_dict in utt["retrieved_passages"]
                                               for k, v in topic_dict.items()],
                        'retrieved_topics': utt["retrieved_topics"],
                    }

                    for topic_dict in utt["retrieved_passages"]:
                        for k, v in topic_dict.items():
                            if k not in wow_hard_knowledge:
                                wow_hard_knowledge[k] = [s.strip() for s in v]

    return wow_knowledge, wow_hard_knowledge


def analyze_schema(file_path='data/wizard_of_wikipedia/train.json'):
    # Function to extract the "schema" of a dictionary (the set of keys)
    def extract_schema(d):
        if isinstance(d, dict):
            speaker = d['speaker'][2:]
            d[speaker] = True
            return frozenset(d.keys())
        return None

    # Analyze the JSON file and group dictionaries by their schema
    def analyze_json_structure(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Assuming the JSON file is a list of dictionaries
        schema_groups = defaultdict(list)

        for i, diag in enumerate(data):
            for item in diag["dialog"]:
                schema = extract_schema(item)
                if schema:
                    schema_groups[schema].append(i)

        return schema_groups

    schemas = analyze_json_structure(file_path)

    # Print the unique schemas and the number of dictionaries using each schema
    for schema, indices in schemas.items():
        print(f"Schema {sorted(list(schema))}: {len(indices)} dictionaries")


if __name__ == '__main__':
    import json
    from collections import defaultdict

    wow_knowledge, wow_hard_knowledge = load_wow_knowledge()
    # print_dialogue('data/wizard_of_wikipedia/data.json', 10)
    for split in ['test', 'train', 'valid']:
        print(f"Processing {split}:")

        with open(f'data/FaithDial/{split}.json', 'r') as file:
            sadf = create_ORFaithDial(file, wow_knowledge, wow_hard_knowledge)

        with open(f'data/FaithDialOR/{split}.json', 'w') as file:
            json.dump(sadf, file, indent=4)

        print()
