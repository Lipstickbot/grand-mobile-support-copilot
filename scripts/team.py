"""Dependency-free native Codex team launcher and verification (Python 3.11+)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / '.team-verification'
ROLES = (
    'orchestrator', 'product_strategist', 'system_architect', 'ui_ux_designer',
    'frontend_developer', 'backend_developer', 'ai_engineer', 'database_engineer',
    'research_agent', 'qa_engineer', 'security_engineer', 'devops_engineer',
)


def load_toml(path):
    with path.open('rb') as stream:
        return tomllib.load(stream)


def codex_path():
    result = shutil.which('codex.exe') or shutil.which('codex')
    if not result:
        raise RuntimeError('Codex CLI is not on PATH. Open a Codex terminal or install the official CLI.')
    return result


def run(args, timeout=90):
    result = subprocess.run(args, cwd=ROOT, text=True, encoding='utf-8',
                            errors='replace', capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {args[:4]}\n{result.stderr[-4000:]}\n{result.stdout[-2000:]}')
    return result


def global_hash():
    home = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')))
    path = home / 'config.toml'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def static_verify():
    EVIDENCE.mkdir(exist_ok=True)
    config = load_toml(ROOT / '.codex/config.toml')
    check(config['sandbox_mode'] == 'workspace-write', 'Parent sandbox must be workspace-write.')
    check(config['approval_policy'] == 'on-request', 'Parent approvals must remain on-request.')
    check(config['sandbox_workspace_write']['network_access'] is False, 'Shell network must be restricted.')
    check(config['agents']['enabled'] is True, 'Native agents must be enabled.')
    check(config['agents']['max_concurrent_threads_per_session'] == 3, 'Expected three spawned-agent slots.')
    check(config['agents']['max_depth'] == 1, 'V1 nested delegation must be disabled.')
    actual = {}
    allowed = {'name', 'description', 'developer_instructions', 'model', 'model_reasoning_effort', 'features'}
    for path in sorted((ROOT / '.codex/agents').glob('*.toml')):
        role = load_toml(path)
        check(not (set(role) - allowed), f'Unexpected role fields: {path.name}')
        name = role.get('name')
        check(name in ROLES and name not in actual, f'Unknown or duplicate role: {name}')
        check(path.stem == name, f'Filename/name mismatch: {path.name}')
        check(all(isinstance(role.get(k), str) and role[k].strip() for k in
                  ('name', 'description', 'developer_instructions', 'model', 'model_reasoning_effort')),
              f'Missing required role fields: {name}')
        check(f'DEC6_{name.upper()}_V1' in role['developer_instructions'], f'Missing marker: {name}')
        for feature in ('apps', 'plugins', 'memories', 'request_permissions_tool'):
            check(role['features'].get(feature) is False, f'{name}: capability {feature} must be disabled.')
        actual[name] = role
    check(set(actual) == set(ROLES), 'All twelve native TOML roles are required.')
    for name in ('product_strategist', 'research_agent'):
        check(actual[name]['features'].get('shell_tool') is False, f'{name}: shell must be disabled.')
    required = ('AGENTS.md', 'team/WORKFLOW.md', 'team/HANDOFF.md', 'team/TESTING.md',
                'team/BUILD-PROMPT.md', 'scripts/build-team.ps1')
    for path in required:
        check((ROOT / path).is_file(), f'Missing coordination file: {path}')
    cli = codex_path()
    version = run([cli, '--version']).stdout.strip()
    startup = run([cli, 'app-server', '--strict-config', '--print-startup-config'])
    (EVIDENCE / 'startup.json').write_text(startup.stdout, encoding='utf-8')
    features = run([cli, 'features', 'list']).stdout
    (EVIDENCE / 'features.txt').write_text(features, encoding='utf-8')
    for name, expected in (('multi_agent', 'true'), ('apps', 'false'), ('plugins', 'false')):
        row = next((line.split() for line in features.splitlines() if line.split() and line.split()[0] == name), [])
        check(row and row[-1] == expected, f'Effective feature {name} is not {expected}; inspect project trust.')
    prompt = run([cli, 'debug', 'prompt-input', 'Configuration verification only. Do not build a product.']).stdout
    (EVIDENCE / 'prompt-input.json').write_text(prompt, encoding='utf-8')
    # Inspect the native spawn schema, not role-name mentions in ordinary instructions.
    prompt_json = json.loads(prompt)
    enums = []
    def walk(value):
        if isinstance(value, dict):
            if 'agent_type' in value and isinstance(value['agent_type'], dict):
                field = value['agent_type']
                enums.append(field.get('enum', []))
                # This installed backend lists custom types in the field description.
                enums.append(re.findall(r'^([a-z][a-z0-9_]*): \{', field.get('description', ''), re.M))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(prompt_json)
    check(any(set(ROLES).issubset(set(enum)) for enum in enums),
          'Native spawn tool does not expose all custom roles. Check project trust and CLI compatibility.')
    catalog = run([cli, 'debug', 'models', '--bundled']).stdout
    (EVIDENCE / 'models.json').write_text(catalog, encoding='utf-8')
    models_json = json.loads(catalog)
    models = models_json if isinstance(models_json, list) else models_json.get('models', [])
    by_slug = {model.get('slug'): model for model in models}
    for name, role in actual.items():
        model = by_slug.get(role['model'])
        check(model is not None, f'{name}: model absent from installed catalog.')
        efforts = {entry['effort'] for entry in model.get('supported_reasoning_levels', [])}
        check(role['model_reasoning_effort'] in efforts, f'{name}: unsupported reasoning effort.')
    return {'version': version, 'roles': list(actual), 'native_role_discovery': 'PASS',
            'strict_config': 'PASS', 'model_efforts': 'PASS', 'live_invocation': 'NOT RUN'}


def live_verify(report, replay=False):
    prompt = '''Configuration smoke test only. Do not build a hackathon solution or edit any files.
Use the native spawn_agent tool to invoke each custom agent_type in this exact roster:
''' + ', '.join(ROLES) + '''.
Use batches of three independent agents. Spawn all three before waiting and observe all completions
before the next batch. If close_agent is available, close completed threads. Otherwise completed
turns release active slots; use the available collaboration list/wait tools and respect runtime limits.
Give each child only this task: "This is a configuration smoke test. Use no tools and do not delegate.
Return one JSON object with role (your configured name), marker (the Role verification marker from
your own developer instructions), and status set to ok. No extra text."
Do not tell children their expected marker. Do not replace custom types with default/worker personas.
Use only collaboration tools yourself. If a role fails, report the actual error instead of simulating it.
Collect all twelve child responses. Your final answer must be JSON with key roles containing those
twelve returned objects. No code fences. Do not claim success for missing or failed children.
'''
    started = time.time()
    if replay:
        events_path = EVIDENCE / 'live-rpc-events.jsonl'
        check(events_path.is_file(), 'No saved app-server trace to verify.')
        print('Validating saved native app-server trace; no new model calls.', flush=True)
    else:
        from trace_native_test import capture
        (EVIDENCE / 'live-prompt.txt').write_text(prompt, encoding='utf-8')
        print('Running read-only native invocation test; twelve roles, batches of three.', flush=True)
        events_path = capture(prompt)
    final_text = (EVIDENCE / 'live-rpc-final.json').read_text(encoding='utf-8').strip()
    if final_text.startswith('```'):
        final_text = final_text.split('\n', 1)[1].rsplit('```', 1)[0]
    final = json.loads(final_text)
    returned = final.get('roles', [])
    check(len(returned) == 12, 'Live test did not return twelve role results.')
    check({r.get('role') for r in returned} == set(ROLES), 'Live role roster mismatch.')
    for row in returned:
        check(row.get('marker') == f'DEC6_{row["role"].upper()}_V1' and row.get('status') == 'ok',
              f'Role instructions were not confirmed: {row}')
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    # Raw app-server events preserve native spawn calls omitted by this CLI JSON adapter.
    raw = []
    seen = set()
    for event in events:
        if event.get('method') != 'rawResponseItem/completed':
            continue
        item = event['params']['item']
        identity = item.get('id')
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        raw.append(item)
    outputs = {item['call_id']: item.get('output', '') for item in raw if item.get('type') == 'function_call_output'}
    spawned = {}
    child_results = {}
    active = set()
    peak = 0
    for item in raw:
        if item.get('type') == 'function_call' and item.get('name') == 'spawn_agent':
            check(item.get('namespace') == 'collaboration', 'Spawn is not a native collaboration call.')
            args = json.loads(item['arguments'])
            role = args.get('agent_type')
            check(role in ROLES, f'Unexpected spawned type: {role}')
            result = json.loads(outputs.get(item['call_id'], '{}'))
            canonical = result.get('task_name')
            check(canonical and canonical.startswith('/root/'), f'Native spawn did not succeed: {role}')
            check(canonical not in spawned, 'Duplicate native task identity.')
            spawned[canonical] = role
            active.add(canonical)
            peak = max(peak, len(active))
        if item.get('type') == 'agent_message' and item.get('author') in spawned and item.get('recipient') == '/root':
            text = ''.join(part.get('text', '') for part in item.get('content', []) if part.get('type') == 'input_text')
            if 'Message Type: FINAL_ANSWER' not in text:
                continue
            payload = text.split('Payload:\n', 1)[1].strip()
            child = json.loads(payload)
            role = spawned[item['author']]
            check(child == {'role': role, 'marker': f'DEC6_{role.upper()}_V1', 'status': 'ok'},
                  f'Native child returned invalid instructions marker: {role}')
            child_results[role] = child
            active.discard(item['author'])
    check(len(spawned) == 12 and set(spawned.values()) == set(ROLES), 'Missing native spawn evidence for all twelve roles.')
    check(set(child_results) == set(ROLES), 'Missing independent native child completion evidence.')
    check(not active, 'Some spawned children never completed.')
    check(2 <= peak <= 3, f'Parallel lifetime evidence outside expected capacity: {peak}')
    report.update(live_invocation='PASS', native_spawn_count=len(spawned),
                  native_child_completions=len(child_results), peak_parallel_children=peak,
                  evidence_sha256=hashlib.sha256(events_path.read_bytes()).hexdigest(),
                  live_evidence_source=str(events_path.relative_to(ROOT)),
                  live_validation_mode='saved trace replay' if replay else 'fresh live run',
                  verification_duration_seconds=round(time.time() - started, 1), child_results=returned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--live', action='store_true')
    verify.add_argument('--replay-trace', action='store_true', help='Validate existing raw evidence without new model calls.')
    build = sub.add_parser('build')
    desc = build.add_mutually_exclusive_group(required=True)
    desc.add_argument('--description')
    desc.add_argument('--description-file', type=Path)
    build.add_argument('--non-interactive', action='store_true')
    build.add_argument('--print-prompt', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        description = args.description if args.description is not None else args.description_file.read_text(encoding='utf-8')
        check(description.strip() and description.strip() != '[PROJECT DESCRIPTION]', 'Supply a real project description.')
        prompt = (ROOT / 'team/BUILD-PROMPT.md').read_text(encoding='utf-8').replace('[PROJECT DESCRIPTION]', description, 1)
        if args.print_prompt:
            print(prompt)
            return 0
        cli = codex_path()
        if args.non_interactive:
            return subprocess.run([cli, 'exec', '--strict-config', '--skip-git-repo-check', '-C', str(ROOT), '-'],
                                  input=prompt, text=True, encoding='utf-8', cwd=ROOT).returncode
        return subprocess.run([cli, '--no-daemon', '--strict-config', '-C', str(ROOT), prompt], cwd=ROOT).returncode
    before = global_hash()
    report = {'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        report.update(static_verify())
        print('PASS: twelve native roles discovered; strict config, permissions and model efforts checked.', flush=True)
        if args.live or args.replay_trace:
            live_verify(report, replay=args.replay_trace)
        report['result'] = 'PASS'
    except Exception as exc:
        report.update(result='FAIL', error=str(exc))
        print(f'FAIL: {exc}', file=sys.stderr)
    report['global_config_unchanged'] = global_hash() == before
    if not report['global_config_unchanged']:
        report.update(result='FAIL', error='Global configuration changed during verification.')
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report.get('live_invocation') == 'PASS':
        (EVIDENCE / 'live-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {EVIDENCE / "verification.json"}')
    return 0 if report['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
