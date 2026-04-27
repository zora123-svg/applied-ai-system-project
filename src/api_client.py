import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


class LastFmAPIError(Exception):
    pass


class LastFmClient:
    API_URL = "http://ws.audioscrobbler.com/2.0/"
    CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "api_cache"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "LastFmClient":
        api_key = os.environ.get("LASTFM_API_KEY")
        if not api_key:
            raise ValueError("Missing LASTFM_API_KEY in environment. Set it in .env or export it.")
        return cls(api_key)

    def _cache_path(self, params: Dict[str, str]) -> Path:
        normalized = json.dumps(params, sort_keys=True)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.CACHE_DIR / f"{digest}.json"

    def _call(self, params: Dict[str, str]) -> Dict:
        request_params = {"api_key": self.api_key, "format": "json", **params}
        cache_path = self._cache_path(request_params)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

        try:
            response = requests.get(self.API_URL, params=request_params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise LastFmAPIError(f"Last.fm request failed: {exc}") from exc

        if isinstance(data, dict) and data.get("error"):
            raise LastFmAPIError(data.get("message", "Last.fm API error"))

        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def search_artist(self, artist_name: str, limit: int = 1) -> List[str]:
        data = self._call({"method": "artist.search", "artist": artist_name, "limit": str(limit)})
        matches = data.get("results", {}).get("artistmatches", {}).get("artist", [])
        if isinstance(matches, dict):
            matches = [matches]
        return [artist.get("name", "") for artist in matches if artist.get("name")][:limit]

    def get_similar_artists(self, artist_name: str, limit: int = 10) -> List[str]:
        data = self._call({"method": "artist.getsimilar", "artist": artist_name, "limit": str(limit)})
        artists = data.get("similarartists", {}).get("artist", [])
        if isinstance(artists, dict):
            artists = [artists]
        return [artist.get("name", "") for artist in artists if artist.get("name")][:limit]

    def get_artist_tags(self, artist_name: str, limit: int = 5) -> List[str]:
        data = self._call({"method": "artist.gettoptags", "artist": artist_name})
        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, dict):
            tags = [tags]
        return [tag.get("name", "") for tag in tags if tag.get("name")][:limit]

    def get_top_artists_by_tag(self, tag: str, limit: int = 10) -> List[str]:
        data = self._call({"method": "tag.gettopartists", "tag": tag, "limit": str(limit)})
        artists = data.get("topartists", {}).get("artist", [])
        if isinstance(artists, dict):
            artists = [artists]
        return [artist.get("name", "") for artist in artists if artist.get("name")][:limit]
