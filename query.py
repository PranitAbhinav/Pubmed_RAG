import boto3, json, pandas as pd
import argparse

from qdrant_client import QdrantClient
from qdrant_client.http import models
from awsauth import qdrantapikey,qdranturl

from awsauth import AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,AWS_SESSION_TOKEN
import boto3
import json
# bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_ACCESS_KEY,aws_session_token=AWS_SESSION_TOKEN)  


def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )



def titan_embed(bedrock,texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        body = {
            "inputText": t,
            # "normalize": True,  # default True for v2 per docs
            # "dimensions": 1024  # v2 supports variable dims incl 1024
        }
        resp = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(body)
        )
        # print(resp)
        vec = json.loads(resp["body"].read())["embedding"]
        out.append(vec)
    return out

'''
hits = client.search(
    collection_name="pubmed_try",
    query_vector=q_vec,
    limit=5
)

for h in hits:
    print(h.payload["PMID"], h.payload["Title"][:150], h.score)
'''
def search_query(query_vec, top_k=3):
    # query_vec = embed_text(query)
    client = QdrantClient(url=qdranturl,api_key=qdrantapikey)
    results = client.search(
        collection_name="pubmed_try",
        query_vector=query_vec,
        limit=top_k,
    )
    return [(hit.payload["PMID"], hit.payload["paragraphs"]) for hit in results]


def summarize_with_bedrock(bedrock, client, query: str, collection: str, top_k: int = 5):
    query_vec = titan_embed(bedrock, [query])[0]
    docs = search_query(query_vec, top_k=5)
    context = "\n".join([d[1] for d in docs])
    prompt =  f"""
    \n\nHuman:
    You are a medical research assistant.
    Use the following research paper extracts to answer the question.
    Context:
    {context}

    Question: {query}

    Summarized Answer (concise, medical-accurate):
    \n\nAssistant:
    """

    response = bedrock.invoke_model(
        modelId="anthropic.claude-v2",
        body=json.dumps({"prompt": prompt, "max_tokens_to_sample": 400})
    )
    return json.loads(response["body"].read())["completion"]


# query = "What are the latest findings about nab-paclitaxel therapy?"
# query_vec = titan_embed([query])[0]

# print(summarize_with_bedrock(query_vec))


def main(args):
    bedrock = get_bedrock_client()
    client = QdrantClient(url=qdranturl, api_key=qdrantapikey)

    summary = summarize_with_bedrock(
        bedrock, client, query=args.query, collection=args.collection, top_k=args.top_k
    )

    print("\n=== Query ===\n\n")
    print(args.query)
    print("\n=== Response ===")
    print(summary)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search PubMed chunks in Qdrant and summarize with Bedrock")
    parser.add_argument("--query", required=True, help="The research question to ask")
    parser.add_argument("--collection", default="pubmed_try", help="Qdrant collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")

    args = parser.parse_args()
    main(args)