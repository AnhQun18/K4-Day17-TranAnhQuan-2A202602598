"""Interactive memory demo. Offline by default; --live explicitly enables API calls."""
from __future__ import annotations

import argparse
import sys
from uuid import uuid4

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--agent", choices=["baseline", "advanced"], default="advanced")
    parser.add_argument("--user", default="demo")
    parser.add_argument("--message", help="Send one message and exit.")
    args = parser.parse_args()
    try:
        config = load_config()
        agent_class = AdvancedAgent if args.agent == "advanced" else BaselineAgent
        agent = agent_class(config, force_offline=not args.live)
        print(f"Mode: {'LIVE' if args.live else 'OFFLINE'} | Agent: {args.agent} | User: {args.user}")
        if args.live:
            print(f"Provider: {config.model.provider} | Model: {config.model.model_name}")
        thread = uuid4().hex
        if args.message:
            print(agent.reply(args.user, thread, args.message)["response"])
            return
        print("/new: thread mới, /quit: thoát. Advanced lưu profile qua lần chạy sau.")
        while True:
            message = input("Bạn: ").strip()
            if message == "/quit":
                break
            if message == "/new":
                thread = uuid4().hex
                print("Đã tạo thread mới.")
                continue
            if message:
                print(agent.reply(args.user, thread, message)["response"])
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
