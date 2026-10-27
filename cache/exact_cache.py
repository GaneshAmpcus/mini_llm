from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import redis

from config import get_settings


logger = logging.getLogger(__name__)


class ExactCache:
    def __init__(self, redis_url: str, ttl_seconds: int, model_name: str, prompt_template_version: str) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.model_name = model_name
        self.prompt_template_version = prompt_template_version
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        logger.debug("Initialized exact cache")

    @classmethod
    def from_settings(cls) -> "ExactCache":
        settings = get_settings()
        return cls(
            redis_url=settings.REDIS_URL,
            ttl_seconds=settings.CACHE_TTL_SECONDS,
            model_name=settings.GROQ_MODEL,
            prompt_template_version=settings.PROMPT_TEMPLATE_VERSION,
        )

    @staticmethod
    def normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question or "").strip().lower()

    @staticmethod
    def build_cache_key(question: str, model_name: str, prompt_template_version: str, *, temperature: float, max_tokens: int) -> str:
        normalized = ExactCache.normalize_question(question)
        payload = {
            "question": normalized,
            "model": model_name,
            "prompt_template_version": prompt_template_version,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return f"mini_llm:exact:{digest}"

    def get(self, question: str) -> str | None:
        try:
            settings = get_settings()
            key = self.build_cache_key(
                question,
                self.model_name,
                self.prompt_template_version,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            raw_value = self._client.get(key)
            logger.debug(
                "Exact cache lookup completed (key=%s, found=%s, question_length=%d)",
                key,
                raw_value is not None,
                len(question),
            )

            if raw_value is None:
                logger.info("Exact cache miss (key=%s)", key)
                return None
            payload = json.loads(raw_value)
            logger.info("Exact cache hit (key=%s)", key)
            return payload.get("answer")
        except Exception:
            logger.exception("Failed to retrieve exact cache entry")
            return None


    def set(self, question: str, answer: str) -> bool:
        try:
            settings = get_settings()
            key = self.build_cache_key(
                question,
                self.model_name,
                self.prompt_template_version,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            payload = {
                "answer": answer,
                "question": self.normalize_question(question),
                "model": self.model_name,
                "prompt_template_version": self.prompt_template_version,
                "stored_at": time.time(),
            }
            self._client.set(key, json.dumps(payload), ex=self.ttl_seconds)
            logger.info("Stored exact cache entry (key=%s, ttl_seconds=%d)", key, self.ttl_seconds)
            return True
        except Exception:
            logger.exception("Failed to store exact cache entry")
            return False

    def invalidate(self, question: str) -> bool:
        try:
            settings = get_settings()
            key = self.build_cache_key(
                question,
                self.model_name,
                self.prompt_template_version,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            deleted = bool(self._client.delete(key))
            logger.info("Invalidated exact cache entry (key=%s, deleted=%s)", key, deleted)
            return deleted
        except Exception:
            logger.exception("Failed to invalidate exact cache entry")
            return False

    def invalidate_all(self) -> bool:
        try:
            keys = list(self._client.scan_iter(match="mini_llm:exact:*"))
            if not keys:
                logger.debug("No exact cache entries found to invalidate")
                return True
            deleted = bool(self._client.delete(*keys))
            logger.info("Invalidated exact cache entries (count=%d, deleted=%s)", len(keys), deleted)
            return deleted
        except Exception:
            logger.exception("Failed to invalidate all exact cache entries")
            return False

    def ping(self) -> bool:
        try:
            available = bool(self._client.ping())
            logger.debug("Exact cache ping completed (available=%s)", available)
            return available
        except Exception:
            logger.exception("Failed to ping exact cache")
            return False
