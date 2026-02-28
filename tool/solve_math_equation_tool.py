"""Tool for solving mathematical equations using WolframAlpha."""

import os
import traceback
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree as ET
from urllib.error import HTTPError
import threading

from tool.base_tool import BasicTool, register_tool

try:
    import wolframalpha

    WOLFRAM_AVAILABLE = True
except ImportError:
    WOLFRAM_AVAILABLE = False

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@register_tool(name="solve_math_equation")
class SolveMathEquationTool(BasicTool):
    """Solve math equations/problems with WolframAlpha."""

    name = "solve_math_equation"
    description_en = (
        "Solve mathematical equations and problems using WolframAlpha. "
        "Supports algebra, calculus, and symbolic/numeric math queries."
    )
    description_zh = (
        "使用 WolframAlpha 求解数学方程和数学问题，"
        "支持代数、微积分与符号/数值计算查询。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A mathematical question or equation to solve",
            }
        },
        "required": ["query"],
    }
    example = '{"query": "x^2 + 2x + 1 = 0, what is x?"}'

    # Class-level key rotation index (shared across instances for load balancing)
    _key_index = 0
    _key_lock = threading.Lock()

    @staticmethod
    def _load_wolfram_keys_from_env() -> List[str]:
        """
        Load Wolfram key(s) from WOLFRAM_ALPHA_API_KEYS env var (comma-separated).
        Falls back to .env files if not found in environment.
        """
        keys = []

        api_keys_str = os.getenv("WOLFRAM_ALPHA_API_KEYS", "").strip()
        if api_keys_str:
            for k in api_keys_str.split(","):
                k = k.strip()
                if k and k not in keys:
                    keys.append(k)
            return keys

        # Try loading from .env files
        if not DOTENV_AVAILABLE:
            return keys

        project_root = Path(__file__).resolve().parent.parent
        candidate_env_files = [
            project_root / "api" / "utils" / "keys.env",
            project_root / "keys.env",
            project_root / ".env",
        ]
        for env_path in candidate_env_files:
            if env_path.exists():
                load_dotenv(env_path, override=False)
                api_keys_str = os.getenv("WOLFRAM_ALPHA_API_KEYS", "").strip()
                if api_keys_str:
                    for k in api_keys_str.split(","):
                        k = k.strip()
                        if k and k not in keys:
                            keys.append(k)
                if keys:
                    break

        return keys

    @classmethod
    def _get_next_key(cls, keys: List[str]) -> str:
        """Get the next API key using round-robin rotation (thread-safe)."""
        if not keys:
            return ""
        with cls._key_lock:
            key = keys[cls._key_index % len(keys)]
            cls._key_index = (cls._key_index + 1) % len(keys)
            return key

    def __init__(self, cfg=None, use_zh=False):
        super().__init__(cfg, use_zh)
        self.client = None
        self.api_keys: List[str] = []
        self.max_retries = 3

        # Load keys from config first
        if cfg is not None:
            cfg_keys = cfg.get("wolfram_api_keys", [])
            if isinstance(cfg_keys, str):
                cfg_keys = [k.strip() for k in cfg_keys.split(",") if k.strip()]
            elif isinstance(cfg_keys, list):
                cfg_keys = [k.strip() for k in cfg_keys if isinstance(k, str) and k.strip()]
            self.api_keys = cfg_keys
            self.max_retries = cfg.get("wolfram_max_retries", 3)

        # Fallback to environment variables
        if not self.api_keys:
            self.api_keys = self._load_wolfram_keys_from_env()

        # Initialize client with first available key
        if WOLFRAM_AVAILABLE and self.api_keys:
            self.client = wolframalpha.Client(self.api_keys[0])

    def _create_client_with_key(self, key: str):
        """Create a new wolframalpha client with the specified key."""
        if WOLFRAM_AVAILABLE and key:
            return wolframalpha.Client(key)
        return None

    @staticmethod
    def _to_bool(value: Any) -> bool:
        """Convert Wolfram success flag to bool safely."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    @staticmethod
    def _normalize_pods(raw_pods: Any) -> List[Dict]:
        if raw_pods is None:
            return []
        if isinstance(raw_pods, list):
            return [p for p in raw_pods if isinstance(p, dict)]
        if isinstance(raw_pods, dict):
            return [raw_pods]
        return []

    @staticmethod
    def _looks_like_interpretation(text: str) -> bool:
        """Heuristic: filter out input-interpretation-like text."""
        if not text:
            return True
        t = text.strip().lower()
        patterns = (
            "input interpretation",
            "solve ",
            " for x",
            " for y",
            " for z",
        )
        return any(p in t for p in patterns)

    @staticmethod
    def _parse_xml_response(xml_text: str) -> Dict:
        """Parse Wolfram XML response to tool result."""
        root = ET.fromstring(xml_text)
        success_raw = root.attrib.get("success", "false")
        success = str(success_raw).lower() == "true"
        if not success:
            return {"error": "Your Wolfram query is invalid. Please try a new query."}

        answer = ""
        for pod in root.findall("pod"):
            title = pod.attrib.get("title", "")
            if title == "Solution":
                subpods = pod.findall("subpod")
                if subpods:
                    plaintext = subpods[0].findtext("plaintext") or ""
                    answer = plaintext
            if title in {"Results", "Solutions"}:
                subpods = pod.findall("subpod")
                for i, sub in enumerate(subpods):
                    text = sub.findtext("plaintext") or ""
                    answer += f"ans {i}: {text}\n"
                break

        if not answer:
            preferred_titles = {
                "Result",
                "Results",
                "Solution",
                "Solutions",
                "Exact result",
                "Decimal approximation",
            }
            for pod in root.findall("pod"):
                title = pod.attrib.get("title", "")
                if title not in preferred_titles:
                    continue
                sub = pod.find("subpod")
                if sub is not None:
                    text = (sub.findtext("plaintext") or "").strip()
                    if text and not SolveMathEquationTool._looks_like_interpretation(text):
                        answer = text
                        break

        if not answer or SolveMathEquationTool._looks_like_interpretation(answer):
            return {"error": "No good Wolfram Alpha result was found."}
        return {"result": answer.strip()}

    @staticmethod
    def _is_retryable_error(error_msg: str) -> bool:
        """Check if the error is retryable with a different key."""
        error_lower = error_msg.lower()
        retryable_patterns = ["rate limit", "quota", "too many requests", "403", "429"]
        return any(p in error_lower for p in retryable_patterns)

    def _query_via_http(self, query: str, api_key: str) -> Dict:
        """Direct HTTP query fallback when wolframalpha SDK fails."""
        try:
            resp = requests.get(
                "https://api.wolframalpha.com/v2/query",
                params={"appid": api_key, "input": query},
                timeout=30,
            )
            if resp.status_code != 200:
                return {
                    "error": (
                        f"WolframAlpha HTTP error: code={resp.status_code}, "
                        f"body_preview={resp.text[:240]}"
                    ),
                    "_retryable": resp.status_code in (403, 429),
                }
            return self._parse_xml_response(resp.text)
        except Exception as e:
            return {
                "error": f"HTTP request failed: {type(e).__name__}: {e}",
                "_retryable": self._is_retryable_error(str(e)),
            }

    def _process_sdk_response(self, res) -> Dict:
        """Process the response from wolframalpha SDK."""
        try:
            success = self._to_bool(res["@success"])
        except Exception:
            success = False

        if not success:
            return {"error": "Your Wolfram query is invalid. Please try a new query."}

        answer = ""
        pods = self._normalize_pods(res["pod"] if "pod" in res else None)
        for pod in pods:
            title = pod.get("@title", "")
            if title == "Solution":
                subpod = pod.get("subpod", {})
                if isinstance(subpod, list):
                    answer = subpod[0].get("plaintext", "") if subpod else ""
                elif isinstance(subpod, dict):
                    answer = subpod.get("plaintext", "")
            if title in {"Results", "Solutions"}:
                subpods = pod.get("subpod", [])
                if isinstance(subpods, dict):
                    subpods = [subpods]
                if isinstance(subpods, list):
                    for i, sub in enumerate(subpods):
                        if isinstance(sub, dict):
                            text = sub.get("plaintext", "")
                            answer += f"ans {i}: {text}\n"
                break

        if not answer:
            try:
                answer = next(res.results).text
            except Exception:
                answer = ""

        if not answer or self._looks_like_interpretation(answer):
            return {"error": "No good Wolfram Alpha result was found."}

        return {"result": answer.strip()}

    def _query_with_retry(self, query: str) -> Dict:
        """Execute query with automatic key rotation on retryable failures."""
        tried_keys = set()
        last_error = None

        for _ in range(min(self.max_retries, len(self.api_keys))):
            current_key = self._get_next_key(self.api_keys)
            if current_key in tried_keys:
                continue
            tried_keys.add(current_key)

            # Try SDK first
            client = self._create_client_with_key(current_key)
            if client:
                try:
                    res = client.query(query)
                    return self._process_sdk_response(res)
                except AssertionError:
                    # SDK assertion error, try HTTP fallback
                    result = self._query_via_http(query, current_key)
                    if "error" not in result or not result.pop("_retryable", False):
                        return result
                    last_error = result
                except HTTPError as e:
                    code = getattr(e, "code", None)
                    if code not in (403, 429):
                        return {"error": f"WolframAlpha HTTP error: code={code}"}
                    last_error = {"error": str(e)}
                except Exception as e:
                    if not self._is_retryable_error(str(e)):
                        return {"error": f"Error querying Wolfram Alpha: {e}"}
                    last_error = {"error": str(e)}
            else:
                result = self._query_via_http(query, current_key)
                if "error" not in result or not result.pop("_retryable", False):
                    return result
                last_error = result

        last_error.pop("_retryable", None)
        return last_error or {"error": "All API keys exhausted."}

    def call(self, params: Union[str, Dict]) -> Dict:
        """Execute WolframAlpha query and return parsed result."""
        params_dict = self.parse_params(params)
        query = params_dict["query"].strip()

        if not query:
            return {"error": "Query cannot be empty."}

        if not WOLFRAM_AVAILABLE:
            return {"error": "wolframalpha package not installed."}

        if not self.api_keys:
            return {
                "error": (
                    "WOLFRAM_ALPHA_API_KEYS is not set. "
                    "Set it in config or environment (.env / keys.env)."
                )
            }

        return self._query_with_retry(query)