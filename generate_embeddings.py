import boto3, json, pandas as pd
import generate_embeddings
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from awsauth import qdrantapikey,qdranturl
from awsauth import AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,AWS_SESSION_TOKEN
import re
import uuid
import tqdm

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_ACCESS_KEY,aws_session_token=AWS_SESSION_TOKEN)  

def titan_embed(texts: list[str]) -> list[list[float]]:
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
        print(resp)
        vec = json.loads(resp["body"].read())["embedding"]
        out.append(vec)
    return out

def chunk_text(text, max_words=200):
    """Split long text into ~200 word chunks."""
    words = re.split(r"\s+", str(text))
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

qdrant = QdrantClient(url=qdranturl,api_key=qdrantapikey)

qdrant.recreate_collection(
    collection_name="pubmed_chunks",
    vectors_config=models.VectorParams(
        size=1024,  # Titan v2 default dim is 1024 if you set it
        distance=models.Distance.COSINE
    )
)

points = []


df=pd.read_csv("pubmed_with_paragraphs.csv")



for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    pmcid = row.get("PMCID", f"no_pmcid_{idx}")
    title = row.get("Title", "")
    paragraph = row.get("paragraphs", "")

    for chunk in chunk_text(paragraph, max_words=500):
        if not chunk.strip():
            continue

        # Embed with Titan v2
        embedding = titan_embed([chunk])[0]  # first (and only) vector

        # Build Qdrant point
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "pmcid": pmcid,
                "title": title,
                "chunk": chunk
            }
        ))

# Upsert into Qdrant
# if points:
#     qdrant.upsert(collection_name="pubmed_try", points=points)


BATCH_SIZE = 50
for i in tqdm.tqdm(range(0, len(points), BATCH_SIZE), desc="Uploading to Qdrant"):
    batch = points[i:i+BATCH_SIZE]
    try:
        qdrant.upsert(collection_name="pubmed_chunks", points=batch)
    except Exception as e:
        pass
