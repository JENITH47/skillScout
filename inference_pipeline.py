import os
import joblib
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Iterable




CAT_MODEL_PATH = "stack_cat.pkl"
XGB_MODEL_PATH = "stack_xgb.pkl"
LGB_MODEL_PATH = "stack_lgb.pkl"
META_MODEL_PATH = "stack_meta.pkl"  


PCA_RESUME_PATH = "pca_resume.pkl"
PCA_JD_PATH = "pca_jd.pkl"
TIER_ENCODER_PATH = "tier_encoder.pkl"


STRUCTURED_COLS = [
    "tier_encoded",
    "num_skills",
    "cs_skill_count",
    "data_skill_count",
    "education_level",
    "experience_years",
    "required_experience_years",
    "experience_gap",
    "micro_match_weight",
    "just_exp_penalty",
    "just_missing_skill_count",
    "just_domain_penalty",
    "bert_sim_score",
    "pca_cos_sim",    
    "pca_l2_dist",
]


def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / (denom + eps)


def safe_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    return np.linalg.norm(a - b, axis=1)


class InferencePipeline:
    def __init__(self,
                 cat_path=CAT_MODEL_PATH,
                 xgb_path=XGB_MODEL_PATH,
                 lgb_path=LGB_MODEL_PATH,
                 meta_path=META_MODEL_PATH,
                 pca_resume_path=PCA_RESUME_PATH,
                 pca_jd_path=PCA_JD_PATH,
                 tier_encoder_path=TIER_ENCODER_PATH):


        for p in [cat_path, xgb_path, lgb_path, meta_path, pca_resume_path, pca_jd_path, tier_encoder_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file missing: {p}")

        
        print("Loading base models...")
        self.cat = joblib.load(cat_path)
        self.xgb = joblib.load(xgb_path)
        self.lgb = joblib.load(lgb_path)
        
        
        print("Loading meta-model...")
        self.meta_model = joblib.load(meta_path)

        
        self.pca_resume = joblib.load(pca_resume_path)
        self.pca_jd = joblib.load(pca_jd_path)
        self.tier_encoder = joblib.load(tier_encoder_path)
        
        print("DEBUG: Stacking Pipeline Initialized Successfully.")

    
    def _build_features(self,
                        resume_emb: Iterable[float],
                        jd_emb: Iterable[float],
                        structured_dict: Dict[str, float]) -> np.ndarray:

        resume_arr = np.asarray(resume_emb, dtype=np.float32).reshape(1, -1)
        jd_arr = np.asarray(jd_emb, dtype=np.float32).reshape(1, -1)

        
        resume_pca = self.pca_resume.transform(resume_arr)
        jd_pca = self.pca_jd.transform(jd_arr)

        
        cos_sim = safe_cosine(resume_pca, jd_pca).reshape(1, 1)
        l2_dist = safe_l2(resume_pca, jd_pca).reshape(1, 1)

        
        struct_vals = []

        
        if "tier_encoded" in structured_dict:
            tier_val = int(structured_dict["tier_encoded"])
        elif "college_tier" in structured_dict:
            try:
                tier_val = int(self.tier_encoder.transform([structured_dict["college_tier"]])[0])
            except Exception:
                tier_val = 0
        else:
            tier_val = 0

        
        for key in STRUCTURED_COLS:
            if key == "tier_encoded":
                struct_vals.append(tier_val)
            elif key == "pca_cos_sim":
                struct_vals.append(float(cos_sim))
            elif key == "pca_l2_dist":
                struct_vals.append(float(l2_dist))
            else:
                struct_vals.append(float(structured_dict.get(key, 0.0)))

        struct_arr = np.asarray(struct_vals, dtype=np.float32).reshape(1, -1)


        X = np.hstack([struct_arr, resume_pca, jd_pca])
        return X


    def predict_single(self, resume_emb, jd_emb, structured_dict) -> float:
        try:
            X = self._build_features(resume_emb, jd_emb, structured_dict)

            p1 = self.cat.predict(X)
            p2 = self.xgb.predict(X)
            p3 = self.lgb.predict(X)

            meta_input = np.column_stack([p1, p2, p3])
            final_score = self.meta_model.predict(meta_input)

            if isinstance(final_score, (list, np.ndarray)):
                score = float(final_score[0])
            else:
                score = float(final_score)

            # Clamp to valid range [0, 10]
            return max(0.0, min(10.0, score))
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

    
    def predict_df(self, df: pd.DataFrame,
                   resume_prefix="resume_emb_",
                   jd_prefix="jd_emb_") -> pd.Series:

        preds = []

        
        resume_cols = sorted([c for c in df.columns if c.startswith(resume_prefix)],
                             key=lambda x: int(x.replace(resume_prefix, "")))
        jd_cols = sorted([c for c in df.columns if c.startswith(jd_prefix)],
                         key=lambda x: int(x.replace(jd_prefix, "")))

        for _, row in df.iterrows():
            resume_emb = row[resume_cols].astype(float).values
            jd_emb = row[jd_cols].astype(float).values

            struct_dict = {}

            if "college_tier" in row.index:
                struct_dict["college_tier"] = row["college_tier"]
            if "tier_encoded" in row.index:
                struct_dict["tier_encoded"] = row["tier_encoded"]

            for key in STRUCTURED_COLS:
                if key not in ["tier_encoded", "pca_cos_sim", "pca_l2_dist"]:
                    struct_dict[key] = row.get(key, 0.0)

            preds.append(self.predict_single(resume_emb, jd_emb, struct_dict))

        return pd.Series(preds)


if __name__ == "__main__":
    
    try:
        pipe = InferencePipeline()
        print("SUCCESS: Pipeline is ready for Stacking Inference.")
    except Exception as e:
        print(f"ERROR: {e}")