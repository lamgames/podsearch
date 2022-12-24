from flask import Flask, request, jsonify
import pandas as pd
import pickle
import util

import cohere
co = cohere.Client('5weCO8zrJmMZhm4sqfYXOg0VpiQhM316hi2cMWUL')

app = Flask(__name__)

response_dict = {}

data = pd.read_csv('./mckinsey_podcasts.csv',index_col=[0])
podcast_titles = data['title']
podcast_intros = data['intro']

flat_parsed_embed = pickle.load(open('flat_parsed_embed.pkl','rb'))
flat_parsed_podcast = pickle.load(open('flat_parsed_podcast.pkl','rb'))

all_pod_queries = pickle.load(open('./podcast_queries.pkl', 'rb'))
queries, query_embs = pickle.load(open('./query.pkl', 'rb'))
query_id = {j:i for i,j in enumerate(queries)}

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/podcast',methods=['POST', "GET"])
def podcast():
    pid = int(request.args.get('pid'))
    podcast = [j[0] for i,j in enumerate(flat_parsed_podcast) if j[1]==pid]
    data = {'podcast': podcast}
    return jsonify(data)


@app.route('/toppodcasts',methods=['POST', "GET"])
def get_top_podcasts():
    query = request.args.get('query')
    topk = int(request.args.get('n'))    
    top_podids = util.get_podcast(co, query, flat_parsed_embed, flat_parsed_podcast, topk)
    top_podcast = [f'<a href=/podcast?pid={i}>{podcast_titles[i]}</a><br><p>{podcast_intros[i]}</p>' for i in top_podids]    
    return '<br><br>'.join(top_podcast)

@app.route('/topsegments',methods=['POST', "GET"])
def get_top_segments():
    query = request.args.get('query')
    topk = int(request.args.get('n'))
    pid = int(request.args.get('pid'))

    top_segments = util.get_answer(co, query, pid, flat_parsed_embed, flat_parsed_podcast, topk)
    return jsonify({'top_segments':top_segments})

@app.route('/summarise',methods=['POST', "GET"])
def summarise():
    data = request.get_json()
    summary = util.summarise(co, data['text'], response_dict)
    return jsonify({'summary': summary})
    

@app.route('/topspans',methods=['POST', "GET"])
def get_top_spans():
    data = request.get_json()
    query = data['query']
    segment = data['segment']
    topk = data['n']
    top_spans = util.get_span(co, query, segment, topk)
    return jsonify({'top_spans':top_spans})

@app.route('/queries',methods=['POST', "GET"])
def get_queries():
    pid = int(request.args.get('pid'))
    q = util.get_queries(pid, all_pod_queries, query_id, query_embs, flat_parsed_podcast, flat_parsed_embed)
    return jsonify({'queries':q})

if __name__ == '__main__':
    app.run(debug=True)


