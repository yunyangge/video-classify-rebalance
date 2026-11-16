"""Whole-tag text normalization and full-dimension embedding.

The query text and pooling match the classification implementation.
"""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from .taxonomy import digest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def clean(value):
    return re.sub('\\s+', ' ', unicodedata.normalize('NFC', str(value or ''))).strip()[:2048]

def useful(value):
    text = clean(value)
    if not text or text.lower() in {'null', 'none', 'untitled', 'unknown', 'video'}:
        return False
    if '�' in text or re.fullmatch('[\\W\\d_]+', text):
        return False
    if re.fullmatch('(?:https?://\\S+|(?:vid(?:eo)?|img|dsc|clip)[_ -]*\\d+)', text, re.I):
        return False
    return True

def evidence(row, source='tags'):
    tags = row.get('tags') or []
    if isinstance(tags, str):
        tags = tags.split(',')
    if source not in {'tags', 'metadata'}:
        raise ValueError('Unknown evidence source')
    values = ([row.get('title'), row.get('description')] if source == 'metadata' else []) + list(tags)
    return list(dict.fromkeys((clean(x) for x in values if useful(x))))

def joined_evidence(row, source='tags'):
    """The exact whole-video text used by the embedding model and lookup cache."""
    return clean(', '.join(evidence(row, source)))

class Encoder:

    def __init__(self, model, device, spec, batch_size=128, max_padded_tokens=16384):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not model.is_dir():
            raise ValueError(f'Model directory not found: {model}')
        (self.torch, self.device, self.spec) = (torch, device, spec)
        (self.batch_size, self.max_padded_tokens) = (batch_size, max_padded_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, padding_side='left')
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
        self.model = AutoModel.from_pretrained(model, local_files_only=True, dtype=dtype, attn_implementation='sdpa').to(device).eval()
        if self.model.config.hidden_size != spec['dimensions']:
            raise ValueError('Model hidden size does not match the configured full embedding dimension')

    def encode(self, texts):
        import numpy as np
        if not texts:
            return np.empty((0, self.spec['dimensions']), dtype=np.float16)
        ids = self.tokenizer([clean(x) for x in texts], truncation=True, max_length=self.spec['max_length'])['input_ids']
        order = np.argsort([len(x) for x in ids], kind='stable')
        (groups, group) = ([], [])
        for index in order:
            index = int(index)
            if group and (len(group) >= self.batch_size or (len(group) + 1) * len(ids[index]) > self.max_padded_tokens):
                groups.append(group)
                group = []
            group.append(index)
        if group:
            groups.append(group)
        result = np.empty((len(texts), self.spec['dimensions']), dtype=np.float16)
        with self.torch.inference_mode():
            for group in groups:
                tokens = self.tokenizer.pad({'input_ids': [ids[i] for i in group]}, padding=True, return_tensors='pt').to(self.device)
                x = self.model(**tokens, use_cache=False).last_hidden_state[:, -1, :].float()
                if x.shape[1] != self.spec['dimensions']:
                    raise ValueError('Embedding dimension mismatch')
                result[group] = self.torch.nn.functional.normalize(x, dim=1).cpu().numpy().astype(np.float16)
        return result

class Matcher:

    def __init__(self, args, config, spec):
        import numpy as np
        (self.np, self.args) = (np, args)
        self.ids = [leaf['id'] for leaf in spec['centers']]
        path = args.run_dir / 'center_vectors.npy'
        meta_path = args.run_dir / 'center_vectors.json'
        meta = json.loads(meta_path.read_text())
        if meta['run_fingerprint'] != digest(spec) or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Stale or changed center vectors')
        self.matrix = np.load(path).astype(np.float32)
        if self.matrix.shape != (len(spec['centers']), config['embedding']['dimensions']):
            raise ValueError('Center shape mismatch')
        if not np.isfinite(self.matrix).all() or (np.linalg.norm(self.matrix, axis=1) == 0).any():
            raise ValueError('Invalid centers')
        self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True)

    def match(self, vectors):
        np = self.np
        x = np.asarray(vectors, dtype=np.float32)
        valid = np.isfinite(x).all(axis=1) & (np.linalg.norm(x, axis=1) > 0)
        x = np.nan_to_num(x)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        similarity = np.clip(x @ self.matrix.T, -1.0, 1.0)
        scores = similarity
        best = scores.argmax(axis=1)
        top = scores[np.arange(len(scores)), best]
        second = np.partition(scores, -2, axis=1)[:, -2]
        margin = top - second
        accepted = valid & (top >= self.args.min_score) & (margin >= self.args.min_margin)
        return (np.where(accepted, best + 1, 0).astype(np.uint16), top, margin)
