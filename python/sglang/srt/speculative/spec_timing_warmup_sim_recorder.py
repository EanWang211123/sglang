import json
from pathlib import Path
from threading import Lock
from typing import Dict, List


class SpecTimingWarmupSimRecorder:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._configs = {
            "draft": {
                "path": self.output_dir / "draft_time.json",
                "bucket_key": "steps",
            },
            "verify": {
                "path": self.output_dir / "verify_time.json",
                "bucket_key": "query-len",
            },
            "draft_extend": {
                "path": self.output_dir / "draft_extend_time.json",
                "bucket_key": "query-len",
            },
        }
        self._data = {
            stage: self._load_stage_data(config["path"])
            for stage, config in self._configs.items()
        }
        for stage in self._configs:
            self._flush_stage(stage)

    def _load_stage_data(self, path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}

        try:
            raw_entries = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        data = {}
        for entry in raw_entries:
            batch_size = int(entry["batch-size"])
            bucket_value = int(entry.get("query-len", entry.get("steps")))
            data[self._make_data_key(batch_size, bucket_value)] = entry
        return data

    def _make_data_key(self, batch_size: int, bucket_value: int) -> str:
        return f"{batch_size}:{bucket_value}"

    def _serialize_stage_data(self, stage: str) -> List[dict]:
        bucket_key = self._configs[stage]["bucket_key"]
        entries = list(self._data[stage].values())
        return sorted(
            [
                {
                    "batch-size": int(entry["batch-size"]),
                    bucket_key: int(entry[bucket_key]),
                    "num": int(entry["num"]),
                    "time": float(entry["time"]),
                }
                for entry in entries
            ],
            key=lambda item: (item["batch-size"], item[bucket_key]),
        )

    def _flush_stage(self, stage: str):
        path = self._configs[stage]["path"]
        path.write_text(
            json.dumps(self._serialize_stage_data(stage), indent=2),
            encoding="utf-8",
        )

    def update(self, stage: str, batch_size: int, bucket_value: int, elapsed_ms: float):
        bucket_key = self._configs[stage]["bucket_key"]
        data_key = self._make_data_key(batch_size, bucket_value)

        with self._lock:
            entry = self._data[stage].get(data_key)
            if entry is None:
                entry = {
                    "batch-size": int(batch_size),
                    bucket_key: int(bucket_value),
                    "num": 0,
                    "time": 0.0,
                }
                self._data[stage][data_key] = entry

            prev_num = int(entry["num"])
            new_num = prev_num + 1
            entry["time"] = (float(entry["time"]) * prev_num + float(elapsed_ms)) / new_num
            entry["num"] = new_num
            self._flush_stage(stage)
