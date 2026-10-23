# Qualitative Error Analysis

## 1. BM25 fails, Dense succeeds

**Query ID:** 239  
**Gold document:** 14079881

> *Cellular aging closely links to an older appearance.*

BM25 retrieves documents containing literal *cellular aging* terminology but misses the gold document in the top 10.

Dense retrieval ranks the gold document **3rd**, successfully connecting *older appearance / perceived age* with biological and molecular ageing.

This illustrates the advantage of semantic retrieval when the query and relevant document use different vocabulary.

## 2. Dense fails, BM25 succeeds

**Query ID:** 72  
**Gold document:** 6076903

> *Activator-inhibitor pairs are provided dorsally by Admpchordin.*

BM25 retrieves the gold document at **rank 9**, whereas Dense misses it in the top 10.

The relevant paper explicitly discusses **ADMP**, **Chordin**, and their opposing signaling relationship.

This case shows that exact domain-specific terminology can still favor lexical retrieval.

## 3. Dense fails, Hybrid succeeds

**Query ID:** 70  
**Gold documents:** 4414547, 5956380

> *Activation of PPM1D suppresses p53 function.*

Dense retrieval returns documents related to general p53 regulation but misses both gold documents.

Hybrid retrieval combines semantic and lexical evidence and retrieves both PPM1D-specific gold documents at **ranks 2 and 6**.

This demonstrates how rank fusion can recover entity-specific evidence missed by dense retrieval alone.

## 4. Cross-Encoder improves Hybrid ranking

**Query ID:** 185  
**Gold document:** 18340282

> *Breast cancer development is determined exclusively by genetic factors.*

Hybrid retrieval identifies the relevant study:

*Gene-environment interactions in 7610 women with breast cancer: prospective evidence from the Million Women Study*

but ranks it only **6th**.

The Cross-Encoder moves the document to **rank 1**.

The study directly examines genetic susceptibility together with established environmental risk factors, making it a strong semantic match for the claim.

This illustrates the role of reranking: the relevant evidence is already present in the candidate set, but joint query-document modeling improves its position substantially.
