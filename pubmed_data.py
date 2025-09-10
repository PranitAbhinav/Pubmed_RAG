import os
import requests
import pandas as pd
from metapub import PubMedFetcher
import requests
from bs4 import BeautifulSoup
import re

def parse_pmc_xml(xml_path):
    """Extract clean paragraphs from a PMC JATS XML file"""
    with open(xml_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml-xml")
    
    paragraphs = []

    # Find <body> content
    body = soup.find("body")
    if not body:
        # print(f"❌ No <body> found in {xml_path}")
        return []

    # Extract paragraphs
    for p in body.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)  # clean up whitespace
        if len(text) > 30:  # skip tiny lines
            paragraphs.append(text)
    
    return paragraphs

def fetch_fulltext_from_pmcid(pmcid, outdir="./papers"):
    os.makedirs(outdir, exist_ok=True)
    url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid.replace('PMC','')}&metadataPrefix=pmc"
    
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code == 200:
        path = os.path.join(outdir, f"{pmcid}.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        # print(f"Full text XML saved for {pmcid} → {path}")
        return path
    else:
        # print(f"Could not fetch XML for {pmcid} (status {resp.status_code})")
        return None
import multiprocessing as mp



path="C:/Users/DLP-I516-216/Downloads/csv-openacces.csv"
df=pd.read_csv(path).head(500)
from tqdm import tqdm
'''
if "paragraphs" not in df.columns:
    df["paragraphs"] = None  
for index,row in tqdm(df.iterrows(),total=len(df)):
    fetch=PubMedFetcher()
    fetch_fulltext_from_pmcid(row["PMCID"],"pdfs/")
    paragraphs = parse_pmc_xml("pdfs/"+row["PMCID"]+".xml")
    os.remove("pdfs/"+row["PMCID"]+".xml")
    df.at[index, "paragraphs"] = paragraphs
'''


def process_pmcid(row):
    """Worker function: fetch XML and parse into paragraphs."""
    pmcid = row["PMCID"]
    try:
        fetch_fulltext_from_pmcid(pmcid, "pdfs/")
        paragraphs = parse_pmc_xml(f"pdfs/{pmcid}.xml")
        if not isinstance(paragraphs, list):
            paragraphs = list(paragraphs)
        return (row.name, paragraphs)  # return index + result
    except Exception as e:
        return (row.name, f"Error for {pmcid}: {e}")

# ---- Multiprocessing setup ---- #
def run_parallel(df, workers=4):
    with mp.Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap(process_pmcid, [row for _, row in df.iterrows()]), 
                            total=len(df), desc="Processing PMCIDs"))

    # update dataframe with results
    for idx, paragraphs in results:
        df.at[idx, "paragraphs"] = paragraphs
    
    return df

if __name__ == "__main__":
    if "paragraphs" not in df.columns:
        df["paragraphs"] = None  

    df = run_parallel(df, workers=mp.cpu_count())

    df.to_csv("pubmed_with_paragraphs.csv")
    print("✅ Completed multiprocessing and saved results")

df.to_csv("pubmed.csv")
