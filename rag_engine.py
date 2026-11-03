# RAG Engine and Scheme Eligibility Module
# Project: AI Voice Assistant for Farmers
# Team: Akash B L (TRV23CE010), Gopika L V (TRV23CE039), Karthika B Nair (TRV23CE042)

import json
import math
import re
from collections import Counter

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class SimpleTFIDFVectorizer:
    """Bilingual unigram + bigram TF-IDF vectorizer and cosine similarity engine."""

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []

    def _tokenize(self, text):
        cleaned = re.sub(r"[^\w\s\u0D00-\u0D7F]", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit_transform(self, corpus):
        tokenized_docs = [self._tokenize(doc) for doc in corpus]
        n_docs = len(tokenized_docs)

        df = Counter()
        for tokens in tokenized_docs:
            for term in set(tokens):
                df[term] += 1

        self.vocab = {term: idx for idx, term in enumerate(df.keys())}
        self.idf = {
            term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for term, count in df.items()
        }

        self.doc_vectors = [self._to_vector(tokens) for tokens in tokenized_docs]
        return self.doc_vectors

    def _to_vector(self, tokens):
        tf = Counter(tokens)
        vec = {}
        norm_sq = 0.0
        for term, count in tf.items():
            if term in self.idf:
                val = count * self.idf[term]
                vec[term] = val
                norm_sq += val * val
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        return {term: val / norm for term, val in vec.items()}

    def cosine_scores(self, query):
        q_tokens = self._tokenize(query)
        q_vec = self._to_vector(q_tokens)
        if not q_vec:
            return [0.0] * len(self.doc_vectors)
        scores = []
        for d_vec in self.doc_vectors:
            dot = sum(val * d_vec.get(term, 0.0) for term, val in q_vec.items())
            scores.append(dot)
        return scores


class KeralaAgriRAGEngine:
    def __init__(self, kb_path="knowledge_base.json"):
        self.kb_path = kb_path
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = data.get("crop_advisories", [])
        self.schemes = data.get("government_schemes", [])
        self.collection = None
        self.vectorizer = SimpleTFIDFVectorizer()

        self.init_vector_db()

    def init_vector_db(self):
        corpus = []
        for doc in self.documents:
            text = (
                doc["crop"] + " " + doc["category"] + " "
                + doc["title_en"] + " " + doc["title_ml"] + " "
                + " ".join(doc.get("keywords", [])) + " "
                + doc["content_en"] + " " + doc["content_ml"]
            )
            corpus.append(text)

        if HAS_CHROMA:
            try:
                client = chromadb.Client()
                self.collection = client.get_or_create_collection(name="kau_agri_docs")
                if self.collection.count() == 0:
                    self.collection.add(
                        ids=[d["id"] for d in self.documents],
                        documents=corpus,
                        metadatas=[
                            {
                                "id": d["id"],
                                "crop": d["crop"],
                                "title_en": d["title_en"],
                                "source_url": d["source_url"]
                            }
                            for d in self.documents
                        ]
                    )
            except Exception:
                self.collection = None

        if len(corpus) > 0:
            self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, top_k=2):
        q_clean = query.lower().strip()
        results = []

        sim_scores = self.vectorizer.cosine_scores(q_clean)

        for i, doc in enumerate(self.documents):
            kw_matches = 0
            for kw in doc.get("keywords", []):
                if kw.lower() in q_clean:
                    kw_matches += 1

            crop_tokens = re.sub(r"[^\w\s\u0D00-\u0D7F]", " ", doc["crop"].lower()).split()
            crop_boost = 0.0
            for ct in crop_tokens:
                if len(ct) > 2 and ct in q_clean:
                    crop_boost += 0.40

            score = (kw_matches * 0.30) + (sim_scores[i] * 0.70) + crop_boost
            item = dict(doc)
            item["similarity_score"] = round(min(0.99, max(0.0, score)), 2)
            results.append((score, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:top_k]]

    def match_schemes(self, crops, land_ha, is_owner=True):
        eligible = []
        for s in self.schemes:
            scheme_crops = [c.lower() for c in s.get("crops", [])]
            has_crop = any(c.lower() in scheme_crops for c in crops) or any("general" in sc for sc in scheme_crops)
            land_valid = s["min_land_ha"] <= land_ha <= s["max_land_ha"]
            owner_valid = True if not s["requires_ownership"] else is_owner

            if has_crop and land_valid and owner_valid:
                eligible.append(s)
        return eligible

    def generate_answer(self, query, retrieved_docs, matched_schemes, language="Bilingual", api_key=""):
        top_doc = retrieved_docs[0]
        top_score = top_doc.get("similarity_score", 0.0)

        if top_score < 0.18:
            output = []
            output.append("### 🌾 കാർഷിക ഉപദേശം (KAU RAG):")
            output.append(
                "നിങ്ങൾ ചോദിച്ച വിഷയത്തിൽ കൃത്യമായ വിവരങ്ങൾ നിലവിലെ ഡാറ്റാബേസിൽ ലഭ്യമല്ല. "
                "കേരള കാർഷിക സർവകലാശാലയുടെ (KAU) മാർഗ്ഗരേഖയിൽ ഉൾപ്പെടുത്തിയിട്ടുള്ള താഴെ പറയുന്ന വിഷയങ്ങളെക്കുറിച്ച് ചോദിക്കുക:\n"
                "- **നെല്ല് (Paddy):** വിരിപ്പ്/മുണ്ടകൻ/പുഞ്ച വിത്തിനങ്ങൾ (ഉമ, ജ്യോതി), മുഞ്ഞ (BPH), തണ്ടുതുരപ്പൻ, ബ്ലാസ്റ്റ്.\n"
                "- **തെങ്ങ് (Coconut):** ചെമ്പൻ ചെല്ലി, കൊമ്പൻ ചെല്ലി, കൂമ്പുചീയൽ (മഹാളി), വളപ്രയോഗം.\n"
                "- **കുരുമുളക് (Pepper):** ദ്രുതവാട്ടം (Quick Wilt), ട്രൈക്കോഡെർമ, ബോർഡോ മിശ്രിതം.\n"
                "- **വാഴ (Banana):** പിണ്ടിപ്പുഴു (Pseudostem weevil), സിഗാട്ടോക്ക ഇലപ്പുള്ളി.\n"
                "- **പച്ചക്കറി & കപ്പ (Vegetables & Tapioca):** കായീച്ച, മൊസൈക് രോഗം.\n"
                "- **മണ്ണ് പരിശോധന & പദ്ധതികൾ:** കുമ്മായം പ്രയോഗം, സുഭിക്ഷ കേരളം, പിഎം കിസാൻ, വിള ഇൻഷുറൻസ്."
            )
            output.append("---")
            output.append("### 🌱 Advisory Notice:")
            output.append(
                f"No close match found in the KAU knowledge base for: *'{query}'*.\n"
                "Please query about Kerala crops (Paddy, Coconut, Pepper, Banana, Vegetables, Tapioca, Soil Acidity) or Government Schemes."
            )
            return "\n\n".join(output)

        context_list = []
        for d in retrieved_docs:
            context_list.append(
                f"Document ID: {d['id']} ({d['source_name']})\n"
                f"English: {d['content_en']}\n"
                f"Malayalam: {d['content_ml']}"
            )
        context_text = "\n\n".join(context_list)

        if api_key and HAS_OPENAI:
            try:
                client = OpenAI(api_key=api_key.strip())
                sys_msg = (
                    "You are a bilingual agricultural assistant for Kerala farmers. "
                    "Answer strictly using the retrieved KAU context below."
                )
                user_msg = f"Question: {query}\nLanguage: {language}\n\nContext:\n{context_text}"
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.2
                )
                return resp.choices[0].message.content
            except Exception:
                pass

        output = []

        if "Malayalam" in language or "Bilingual" in language:
            output.append(f"### കാർഷിക ശുപാർശ (KAU): {top_doc['title_ml']}")
            output.append(top_doc["content_ml"])
            if matched_schemes:
                output.append("**അർഹതയുള്ള സർക്കാർ പദ്ധതികൾ (Eligible Schemes):**")
                for s in matched_schemes[:2]:
                    output.append(f"- **{s['name_ml']}**: {s['benefit_ml']} ([Portal]({s['source_url']}))")

        if "English" in language or "Bilingual" in language:
            output.append(f"### KAU Advisory: {top_doc['title_en']}")
            output.append(top_doc["content_en"])
            if matched_schemes:
                output.append("**Eligible Government Schemes:**")
                for s in matched_schemes[:2]:
                    output.append(f"- **{s['name_en']}**: {s['benefit_en']} ([Portal]({s['source_url']}))")

        output.append(f"**Reference:** [{top_doc['source_name']}]({top_doc['source_url']})")
        return "\n\n".join(output)