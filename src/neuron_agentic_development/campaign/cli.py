"""Strict JSON command-line transport for the campaign controller."""

import argparse
import json
import os
import sys
from pathlib import Path

from .contracts import CampaignError, ContractError
from .controller import CampaignController


def _load(path):
    try:
        if path == "-":
            return json.load(sys.stdin)
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON input {path!r}: {error}")


def _root(value):
    return str(
        Path(value or os.environ.get("NEURON_PROJECT_ROOT") or os.getcwd()).resolve()
    )


def _open(args, campaign_id=None):
    if campaign_id is None:
        campaign_id = getattr(args, "campaign", None)
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ContractError("campaign_id must be a non-empty string")
    return CampaignController.open(
        _root(getattr(args, "project_root", None)),
        campaign_id,
    )


def _add_campaign(parser):
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--project-root")


def _add_key(parser):
    parser.add_argument("--idempotency-key", required=True)


def _parser():
    parser = argparse.ArgumentParser(prog="neuron-campaign")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("init", help="initialize a campaign")
    command.add_argument("--spec", required=True, help="JSON file or - for stdin")

    for name in ("context", "status", "replay", "report"):
        command = commands.add_parser(name)
        _add_campaign(command)

    command = commands.add_parser("prepare")
    command.add_argument("--request", required=True, help="JSON file or - for stdin")
    command.add_argument("--project-root")

    command = commands.add_parser("validate-submission")
    command.add_argument("--submission", required=True)
    command.add_argument("--project-root")

    command = commands.add_parser("format-submission")
    _add_campaign(command)
    _add_key(command)
    command.add_argument("--task", required=True)
    command.add_argument("--facts", required=True)

    command = commands.add_parser("progress")
    command.add_argument("--request", required=True)
    command.add_argument("--project-root")

    command = commands.add_parser("submit")
    command.add_argument("--submission", required=True, help="JSON file or - for stdin")
    command.add_argument("--project-root")

    command = commands.add_parser("pause")
    _add_campaign(command)
    _add_key(command)
    command.add_argument("--mode", choices=("after-task", "now"), required=True)

    command = commands.add_parser("resume")
    _add_campaign(command)
    _add_key(command)

    command = commands.add_parser("cancel")
    _add_campaign(command)
    _add_key(command)
    command.add_argument("--task", required=True)

    command = commands.add_parser("audit")
    _add_campaign(command)
    command.add_argument("--strict", action="store_true")

    command = commands.add_parser("finalize")
    _add_campaign(command)
    _add_key(command)
    return parser


def _dispatch(args):
    if args.command == "init":
        return CampaignController.initialize(_load(args.spec)).context()
    if args.command == "prepare":
        request = _load(args.request)
        if not isinstance(request, dict) or not isinstance(request.get("campaign_id"), str):
            raise ContractError("ActionRequest must contain campaign_id")
        return _open(args, request.get("campaign_id")).prepare(request)
    if args.command in ("submit", "validate-submission"):
        submission = _load(args.submission)
        if not isinstance(submission, dict) or not isinstance(submission.get("campaign_id"), str):
            raise ContractError("AgentSubmission must contain campaign_id")
        controller = _open(args, submission.get("campaign_id"))
        return (controller.submit(submission) if args.command == "submit"
                else controller.validate_submission(submission))
    if args.command == "progress":
        document = _load(args.request)
        if not isinstance(document, dict):
            raise ContractError("Progress must be an object")
        return _open(args, document.get("campaign_id")).progress(document)

    controller = _open(args)
    if args.command == "context":
        return controller.context()
    if args.command == "report":
        return controller.report()
    if args.command == "format-submission":
        return controller.format_submission(args.task, _load(args.facts), args.idempotency_key)
    if args.command == "status":
        return controller.status()
    if args.command == "pause":
        return controller.pause(args.mode, args.idempotency_key)
    if args.command == "resume":
        return controller.resume(args.idempotency_key)
    if args.command == "cancel":
        return controller.cancel(args.task, args.idempotency_key)
    if args.command == "audit":
        return controller.audit()
    if args.command == "finalize":
        return controller.finalize(args.idempotency_key)
    if args.command == "replay":
        return controller.replay()
    raise ContractError(f"unsupported command: {args.command}")


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
        strict_failure = (
            args.command == "audit" and args.strict and not result["ok"]
        ) or (args.command in ("validate-submission", "format-submission") and not result["valid"])
        print(
            json.dumps(
                {"ok": not strict_failure, "result": result},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 3 if strict_failure else 0
    except CampaignError as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": error.__class__.__name__,
                        "message": str(error),
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
