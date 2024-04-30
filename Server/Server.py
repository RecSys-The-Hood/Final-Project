# %%
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from flask import Flask, request, jsonify
import torch
# 1. load model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    
    return "Hello"

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming the incoming data is in JSON format
    data = request.get_json()  # Retrieve JSON data
    docs=data['message']
    sentence=[docs]
    # cluster_embeddings=data['embeddings']
    embeddings = model.encode(sentence)
    # print(embeddings.shape)
    data_embeddings=data['embeddings']
    cluster_embeddings=torch.tensor(list(data_embeddings.values()))

    similarities = cos_sim(embeddings[0], cluster_embeddings)

    dict_similarities={}
    keys=list(data_embeddings.keys())
    for i in range(len(keys)):
        dict_similarities[keys[i]]=similarities[0,i].item()
        
    sorted_by_values_desc = sorted(dict_similarities.items(), key=lambda item: item[1], reverse=True)
    
    # print('similarities:', sorted_by_values_desc)

    return jsonify({'similarities': sorted_by_values_desc})
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 

