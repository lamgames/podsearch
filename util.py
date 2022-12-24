import numpy as np
from nltk.tokenize import sent_tokenize

def generate(co, text, response_dict):
    if text in response_dict: return response_dict[text]
    response = co.generate(
              prompt=text, model='command-xlarge-20221108',max_tokens=200, 
              temperature=0, 
              k=0, 
              p=0.5, 
              frequency_penalty=0, 
              presence_penalty=0, 
              stop_sequences=["--"], 
              return_likelihoods='NONE')
    response_dict[text] = response.generations[0].text
    return response_dict[text]

def get_podcast(co, query, flat_parsed_embed, flat_parsed_podcast, topk=5):
    qemb = np.array(co.embed([query]).embeddings)
    ids = np.argsort(-np.matmul(qemb, flat_parsed_embed.T)[0])
    podcast_ids = []
    vis_podid = set()
    for i in ids:        
        segment, podid = flat_parsed_podcast[i][0], flat_parsed_podcast[i][1]
        if segment[-1] == '?':continue
        if podid not in vis_podid: podcast_ids.append(podid); vis_podid.add(podid)
    return podcast_ids[:topk]


def get_span(co, query, answer, topk=3):
    qemb = np.array(co.embed([query]).embeddings)
    sentences = sent_tokenize(answer)
    sent_emb = np.array(co.embed(sentences).embeddings)
    ids = np.argsort(-np.matmul(qemb, sent_emb.T)[0])
    return [sentences[i] for i in ids][:topk]

def get_answer(co, query, podid, flat_parsed_embed, flat_parsed_podcast, topk=5):
    podids_from_flat = [i for i,j in enumerate(flat_parsed_podcast) if j[1]==podid]
    pod_embs = flat_parsed_embed[podids_from_flat]
    qemb = np.array(co.embed([query]).embeddings)
    ids = np.argsort(-np.matmul(qemb, pod_embs.T)[0])
    return [flat_parsed_podcast[podids_from_flat[i]] for i in ids[:topk]]

def summarise(co, text, response_dict):
    return generate(co, f'Summarise the following passage:\n\n{text}', response_dict)
