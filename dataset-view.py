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


def create_ORFaithDial(split='train'):
    # Replace 'filename.txt' with your actual file name
    wow_knowledge = {}
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

    skipped_because_none = 0
    skipped = 0
    skipped_but_found = 0
    all = 0
    with open(f'data/FaithDial/{split}.json', 'r') as file:
        faith_dial_data = json.load(file)
        for diag in tqdm(faith_dial_data):
            for utt in diag['utterances']:
                all += 1
                knowledge_key = utt["original_response"]
                knowledge_key = fd_fixes_mapping.get(knowledge_key, knowledge_key)

                if knowledge_key is None:
                    knowledge_key = try_find_knowledge_sent(utt["knowledge"], wow_knowledge, diag)
                    
                try:
                    knowledge = wow_knowledge[knowledge_key]
                except KeyError:
                    if knowledge_key is None:
                        skipped_because_none += 1
                    else:
                        skipped += 1
                    continue  # Skip this utterance
                utt['passages'] = knowledge['retrieved_passages']
                utt['topics'] = knowledge['retrieved_topics']
                utt['checked_sentence'] = knowledge['checked_sentence']
                utt['checked_passage'] = knowledge['checked_passage']

                # todo check if utt['knowledge'] is in the retrieved passages

    print(f"{skipped_because_none}+{skipped}/{all} skipped")
    print(f"skipped_because_none + skipped / all skipped")

    with open(f'data/FaithDialOR/{split}.json', 'w') as file:
        json.dump(faith_dial_data, file, indent=4)


if __name__ == '__main__':
    # print_dialogue('data/wizard_of_wikipedia/data.json', 10)
    create_ORFaithDial('train')
