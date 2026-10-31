"""Capture installed app-server protocol evidence for the native team smoke test."""
import json
import queue
import subprocess
import threading
import time

from team import ROOT, EVIDENCE, codex_path, check


def capture(prompt, timeout=600):
    EVIDENCE.mkdir(exist_ok=True)
    path = EVIDENCE / 'live-rpc-events.jsonl'
    inbox = queue.Queue()
    with (EVIDENCE / 'live-rpc-stderr.txt').open('w', encoding='utf-8') as errors:
        process = subprocess.Popen([codex_path(), 'app-server', '--strict-config', '--listen', 'stdio://'],
                                   cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=errors, text=True, encoding='utf-8', errors='replace', bufsize=1)
        def read():
            for line in process.stdout:
                inbox.put(line)
            inbox.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        def send(value):
            process.stdin.write(json.dumps(value) + '\n')
            process.stdin.flush()
        def next_message(log):
            remaining = deadline - time.monotonic()
            check(remaining > 0, 'App-server smoke test timed out.')
            try:
                line = inbox.get(timeout=min(30, remaining))
            except queue.Empty:
                return None
            check(line is not None, 'App-server exited before test completion.')
            value = json.loads(line)
            log.write(json.dumps(value, ensure_ascii=False) + '\n')
            log.flush()
            if 'method' in value and 'id' in value:
                # The smoke test authorizes no tool execution or approvals.
                send({'id': value['id'], 'error': {'code': -32601,
                      'message': 'Smoke test rejects client tool/approval requests.'}})
            return value
        def response(expected, log):
            while True:
                value = next_message(log)
                if value and value.get('id') == expected:
                    check('error' not in value, f'App-server request {expected} failed: {value}')
                    return value['result']
        try:
            with path.open('w', encoding='utf-8') as log:
                send({'id': 1, 'method': 'initialize', 'params': {
                    'clientInfo': {'name': 'dec6_team_verifier', 'version': '1.0.0'},
                    'capabilities': {'experimentalApi': True}}})
                response(1, log)
                send({'method': 'initialized', 'params': {}})
                send({'id': 2, 'method': 'thread/start', 'params': {
                    'cwd': str(ROOT), 'ephemeral': True, 'sandbox': 'read-only',
                    'approvalPolicy': 'on-request', 'experimentalRawEvents': True,
                    'config': {'model_reasoning_effort': 'low'}}})
                thread = response(2, log)
                thread_id = thread['thread']['id']
                send({'id': 3, 'method': 'turn/start', 'params': {
                    'threadId': thread_id, 'input': [{'type': 'text', 'text': prompt}], 'effort': 'low'}})
                response(3, log)
                final_text = None
                while True:
                    value = next_message(log)
                    if not value:
                        continue
                    params = value.get('params', {})
                    if params.get('threadId') != thread_id:
                        continue
                    if value.get('method') == 'item/completed':
                        item = params.get('item', {})
                        if item.get('type') == 'agentMessage':
                            final_text = item.get('text')
                    if value.get('method') == 'turn/completed':
                        check(params['turn']['status'] == 'completed', f'Native smoke turn failed: {params["turn"]}')
                        check(final_text, 'No final agent message received.')
                        (EVIDENCE / 'live-rpc-final.json').write_text(final_text, encoding='utf-8')
                        return path
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == '__main__':
    prompt = (EVIDENCE / 'live-prompt.txt').read_text(encoding='utf-8')
    print(capture(prompt))
