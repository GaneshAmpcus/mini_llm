from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time

import redis

from config import get_settings


logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, redis_url: str, ttl_seconds: int, threshold: float, model_name: str, prompt_template_version: str) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.threshold = threshold
        self.model_name = model_name
        self.prompt_template_version = prompt_template_version
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        logger.debug(
            "Initialized semantic cache (threshold=%s, ttl_seconds=%s, model=%s, prompt_version=%s)",
            threshold,
            ttl_seconds,
            model_name,
            prompt_template_version,
        )

    @classmethod
    def from_settings(cls) -> "SemanticCache":
        settings = get_settings()
        return cls(
            redis_url=settings.REDIS_URL,
            ttl_seconds=settings.CACHE_TTL_SECONDS,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            model_name=settings.GROQ_MODEL,
            prompt_template_version=settings.PROMPT_TEMPLATE_VERSION,
        )

    @staticmethod
    def normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question or "").strip()

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        magnitude_left = math.sqrt(sum(a * a for a in left))
        magnitude_right = math.sqrt(sum(a * a for a in right))
        if magnitude_left == 0 or magnitude_right == 0:
            return 0.0
        return dot / (magnitude_left * magnitude_right)

    @staticmethod
    def build_cache_key(question: str) -> str:
        normalized = SemanticCache.normalize_question(question)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"mini_llm:semantic:{digest}"

    def _generate_embedding(self, question: str) -> list[float] | None:
        try:
            if get_settings().EMBEDDING_PROVIDER != "local":
                raise ValueError("Only the local embedding provider is supported in this baseline implementation.")
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(get_settings().EMBEDDING_MODEL)
            vector = model.encode(question, convert_to_numpy=True)
            embedding = [float(value) for value in vector.tolist()]
            logger.debug("Generated semantic embedding (dimensions=%d)", len(embedding))
            return embedding
        except Exception:
            logger.exception("Failed to generate semantic embedding (question_length=%d)", len(question))
            return None

    def get(self, question: str) -> tuple[str, float] | None:
        settings = get_settings()
        if not settings.ENABLE_SEMANTIC_CACHE:
            logger.debug("Semantic cache lookup skipped because it is disabled")
            return None
        try:
            query_embedding = self._generate_embedding(question)
            if query_embedding is None:
                logger.debug("Semantic cache lookup skipped because embedding generation failed")
                return None
            entries = list(self._client.scan_iter(match="mini_llm:semantic:*"))
            logger.debug("Scanning %d semantic cache entries", len(entries))
            best_match: tuple[str, float] | None = None
            for raw_key in entries:
                payload_raw = self._client.get(raw_key)
                if payload_raw is None:
                    logger.debug("Semantic cache entry disappeared during lookup (key=%s)", raw_key)
                    continue
                payload = json.loads(payload_raw)
                if payload.get("prompt_version") != settings.PROMPT_TEMPLATE_VERSION:
                    logger.debug("Skipping semantic cache entry with different prompt version (key=%s)", raw_key)
                    continue
                if payload.get("model") != settings.GROQ_MODEL:
                    logger.debug("Skipping semantic cache entry with different model (key=%s)", raw_key)
                    continue
                candidate_embedding = payload.get("embedding") or []
                score = self._cosine_similarity(query_embedding, candidate_embedding)
                logger.debug(
                    "Compared semantic cache entry (key=%s, similarity=%.6f, threshold=%.6f)",
                    raw_key,
                    score,
                    self.threshold,
                )
                if score >= self.threshold:
                    if best_match is None or score > best_match[1]:
                        best_match = (payload.get("answer", ""), float(score))
            if best_match is None:
                logger.info("Semantic cache miss (entries_scanned=%d)", len(entries))
            else:
                logger.info(
                    "Semantic cache hit (similarity=%.6f, threshold=%.6f)",
                    best_match[1],
                    self.threshold,
                )
            return best_match
        except Exception:
            logger.exception("Failed to retrieve semantic cache entry")
            return None

    def set(self, question: str, answer: str) -> bool:
        try:
            settings = get_settings()
            embedding = self._generate_embedding(question)
            if embedding is None:
                logger.warning("Semantic cache entry not stored because embedding generation failed")
                return False
            payload = {
                "question": self.normalize_question(question),
                "answer": answer,
                "embedding": embedding,
                "model": settings.GROQ_MODEL,
                "prompt_version": settings.PROMPT_TEMPLATE_VERSION,
                "stored_at": time.time(),
            }
            key = self.build_cache_key(question)
            self._client.set(key, json.dumps(payload), ex=self.ttl_seconds)
            logger.info(
                "Stored semantic cache entry (key=%s, embedding_dimensions=%d, ttl_seconds=%d)",
                key,
                len(embedding),
                self.ttl_seconds,
            )
            return True
        except Exception:
            logger.exception("Failed to store semantic cache entry")
            return False

    def invalidate(self, question: str) -> bool:
        try:
            key = self.build_cache_key(question)
            deleted = bool(self._client.delete(key))
            logger.info("Invalidated semantic cache entry (key=%s, deleted=%s)", key, deleted)
            return deleted
        except Exception:
            logger.exception("Failed to invalidate semantic cache entry")
            return False

    def invalidate_all(self) -> bool:
        try:
            keys = list(self._client.scan_iter(match="mini_llm:semantic:*"))
            if not keys:
                logger.debug("No semantic cache entries found to invalidate")
                return True
            deleted = bool(self._client.delete(*keys))
            logger.info("Invalidated semantic cache entries (count=%d, deleted=%s)", len(keys), deleted)
            return deleted
        except Exception:
            logger.exception("Failed to invalidate all semantic cache entries")
            return False

    def ping(self) -> bool:
        try:
            available = bool(self._client.ping())
            logger.debug("Semantic cache ping completed (available=%s)", available)
            return available
        except Exception:
            logger.exception("Failed to ping semantic cache")
            return False
