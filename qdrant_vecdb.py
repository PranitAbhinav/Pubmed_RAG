import generate_embeddings
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from awsauth import qdrantapikey,qdranturl


client = QdrantClient(url=qdranturl,api_key=qdrantapikey)


df=pd.read_csv("C:/Users/DLP-I516-216/Desktop/Workspace/pubmed_/pubmed_with_paragraphs.csv")
embeddings=generate_embeddings.titan_embed(df['paragraphs'].tolist())

client.recreate_collection(
    collection_name="pubmed_try",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)



points = [
    models.PointStruct(
        id=i,
        vector=embeddings[i],
        payload={"PMID": df.iloc[i]["PMID"], "Title": df.iloc[i]["Title"],"paragraphs": df.iloc[i]["paragraphs"]}
    )
    for i in range(len(df))
]

client.upsert(collection_name="pubmed_try", points=points)
