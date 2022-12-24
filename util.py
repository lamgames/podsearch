import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

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
        if segment[-1] == '?': continue
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
    top_segments = []
    for i in ids:
        if flat_parsed_podcast[podids_from_flat[i]][0][-1]!='?':
            top_segments.append(flat_parsed_podcast[podids_from_flat[i]])
            if len(top_segments)==topk: return top_segments
    
    return top_segments    

def summarise(co, text, response_dict):
    return generate(co, f'Summarise the following passage:\n\n{text}', response_dict)

def get_queries(podid, all_pod_queries, query_id, query_embs, flat_parsed_podcast, flat_parsed_embed):
    cur_pod_queries = [i[0] for i in all_pod_queries if i[1]==podid]
    cur_pod_qid = [query_id[i] for i in cur_pod_queries]
    qembs = query_embs[cur_pod_qid]

    segids_from_flat = [i for i,j in enumerate(flat_parsed_podcast) if j[1]==podid]
    seg_embs = flat_parsed_embed[segids_from_flat]

    cos_sim = cosine_similarity(qembs, seg_embs)
    rows = [int(i) for i in np.argsort(-cos_sim,axis=None)/qembs.shape[0]]
    sel_q = set()
    for i in rows:
        if cur_pod_queries[i][-1]!='?': continue
        sel_q.add(cur_pod_queries[i])
        if len(sel_q)==10: break
    return list(sel_q)
