import json
import random
import os
import openai
import tqdm
from tqdm import trange
import time
import argparse
import logging
import ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 5
REQ_TIME_GAP = 35

openai.api_type = "azure"
openai.api_base = ""
openai.api_version = '2023-03-15-preview'
openai.api_key = ""
                  

def parse_gpt(res):
    domains = []
    summaries = []
    task_types = []
    res_split = res.split("\n\n")[1::2]
    for sp in res_split:
        sp_split = sp.split("\n")
        domain, summary, task_type = sp_split[0], sp_split[1], sp_split[2]

        assert domain.startswith('[domain]')
        assert summary.startswith('[summary]')
        assert task_type.startswith('[task type]')

        domain = domain[8:]
        summary = summary[9:]
        task_type = task_type[11:]

        domains.append(domain)
        summaries.append(summary)
        task_types.append(task_type)
        
    return domains, summaries, task_types


def decoder_gpt(input, temp, engine, max_length):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                engine=engine, # gpt4-32k
                messages=[{
                    "role":"user",
                    "content":input,
                }],
                temperature=temp,
                max_tokens=max_length,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
            content = response['choices'][0]['message']['content']
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"


def write_new_demo(demo_file, single_res, raw_dict):
    conversations = raw_dict[single_res['id']]

    query = ""
    for uttr in conversations:
        if uttr['from'] == "human":
            query += uttr['value']

    new_demo = {
        "user query": query,
        "label": single_res['task_type'],
        "domain": single_res["domain"],
        "summary": single_res["summary"],
    }
    with open(demo_file, 'a+', encoding='utf-8') as f:
        line = json.dumps(new_demo, ensure_ascii=False)
        f.write(line+'\n') 


def demo_selection(demo_file, num_demo):
    query = []
    output = []

    with open(demo_file, 'r', encoding='utf-8') as f:
        for j in f.readlines():
            j = json.loads(j)
            query.append("[user query] " + j['user query'])
            output.append('[domain]' + j['domain'] + '\n[summary]' + j['summary'] + '\n[task type]' + j['label'])
    
    assert len(query) == len(output)
    selected_demos = random.sample([i for i in range(len(query))], num_demo)

    queries = ""
    for q_id in range(len(selected_demos)):
        queries += ("\n\n** Input {} **\n\n".format(q_id) + query[selected_demos[q_id]])
    
    outputs = ""
    for o_id in range(len(selected_demos)):
        outputs += ("\n\n** Output {} **\n\n".format(o_id) + output[selected_demos[o_id]])

    return queries + outputs


def get_json_list(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GPT4-based ShareGPT classification")
    parser.add_argument(
        "--data_file", default="sharegpt_data.json", help="data to be analyzed, in this project, shareGPT"
        )
    parser.add_argument(
        "--demo_file", default="demonstrations.json", help="path to store the demonstration file"
        )
    parser.add_argument(
        "--output_file", default="./shareGPT_merged.json", help="path to store the final classify results"
        )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="temperature for GPT engine"
        )
    parser.add_argument(
        "--frequency", type=float, default=0.05, help="threshold to determine whether a demonstration need to be added"
        )
    parser.add_argument(
        "--max_output_token", type=int, default=800, help="maximum output length for GPT engine"
        )
    parser.add_argument(
        "--step", type=int, default=3, help="parallel steps for iteration"
        )
    parser.add_argument(
        "--engine", default="gpt-4", help="GPT engine to be used"
    )
    args = parser.parse_args()

    # ray.init()

    raw_data = json.load(open(args.data_file, "r"))

    raw_dict = {item['id']: item['conversations'] for item in raw_data}

    # instruction = "Users usually leverage GPT for some real-world applications. Please classify the user queries with respect to task types, as fine-grained as possible. \
    #     Remember that each content could have multiple categories."
    instruction = "You will be given a user query in user-GPT conversation. Please classify the user queries with respect to task types, as fine-grained as possible following:\n(1) Identify specific domain/topic of the user query.\n(2) Give a brief summary of the user query.\n(3) Give task types for the user query. There could be multiple types, and organize them from coarse to fine."

    answer_trigger = "\n\n** What are the task type of the following samples? Please strictly follow the style of previous demonstrations. Make sure you respond to every input user query.**"

    # random.shuffle(raw_data)

    demo_type = []
    type_dict = {}

    # while True:
    # for item_idx, item in enumerate(tqdm.tqdm(raw_data[:10000])):
    for item_idx in trange(0, len(raw_data), args.step):
        
        item = raw_data[item_idx]
        conversations = item['conversations']
        demos = demo_selection(demo_file=args.demo_file, num_demo=args.step)
        text =  instruction + demos + answer_trigger

        step_res = []
        query_total = ""
        for i in range(args.step):
            query = "\n\n** Input {} ** \n\n [user query] ".format(i)
            step_res.append({
                "id": raw_data[item_idx+ i]['id'],
            })
            for uttr in raw_data[item_idx + i]['conversations']:
                if uttr['from'] == "human":
                    query += (uttr['value'] + "\n")
            query_total += (query + "\n\n")
        
        prompt = text + query_total

        res = decoder_gpt(prompt, args.temperature, args.engine, args.max_output_token)
        if res == "error":
            continue

        try:
            domain, summary, task_type = parse_gpt(res)
        except Exception as e:
            logger.error(e)
            continue

        try:
            assert len(domain) == len(summary) == len(task_type) == len(step_res)
        except:
            continue
        
        for res_id in range(len(step_res)):
            step_res[res_id]['domain'] = domain[res_id]
            step_res[res_id]['summary'] = summary[res_id]
            step_res[res_id]['task_type'] = task_type[res_id]

        
        for t_id, t_s in enumerate(task_type):
            for t in t_s.split(", "):
                if t not in type_dict.keys():
                    type_dict[t] = 1
                else:
                    if type_dict[t] >= item_idx * args.step * args.frequency and (t not in demo_type) and (t!="error"):
                        demo_type.append(t)
                        write_new_demo(args.demo_file, step_res[t_id], raw_dict)
                    type_dict[t] += 1

        with open(args.output_file, 'a+', encoding='utf-8') as f:
            for single_res in step_res:
                line = json.dumps(single_res, ensure_ascii=False)
                f.write(line+'\n') 
        
        logger.info(
            f"Waiting for {REQ_TIME_GAP} seconds before sending the next request."
        )
        time.sleep(REQ_TIME_GAP)