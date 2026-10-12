"""Verified payload, bounded foreground processes, final evaluation, own-VM release."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time


def main():
    root = Path(os.environ.get('RECSYS_RUNTIME_ROOT', '/content/qwen_continue'))
    settings = json.loads((root/'launch.json').read_text())
    credentials = root/'drive_credentials.json'
    os.chmod(credentials, 0o600)
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import AuthorizedSession
    client = AuthorizedSession(Credentials.from_authorized_user_file(str(credentials)).with_quota_project(settings['quota_project']))
    headers = {'Accept':'application/json','X-Colab-Client-Agent':'colab-cli'}
    def release():
        # Only the endpoint reserved by this submission may be released.
        url = 'https://colab.research.google.com/tun/m/unassign/' + settings['endpoint']
        result = client.get(url, params={'authuser':'0'}, headers=headers, timeout=30)
        if result.status_code == 404:
            return
        result.raise_for_status()
        text = result.text.removeprefix(")]}'\n")
        token = json.loads(text)['token']
        result = client.post(url, params={'authuser':'0'}, headers={**headers,'X-Goog-Colab-Token':token}, timeout=30)
        if result.status_code != 404:
            result.raise_for_status()

    def stream(command, deadline):
        env = dict(os.environ, USE_TF='0', USE_TORCH='1', HNSWLIB_NO_NATIVE='1', TOKENIZERS_PARALLELISM='false')
        child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
        def stop():
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=240)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
        timer = threading.Timer(max(1,deadline-time.time()),stop)
        timer.daemon=True
        timer.start()
        try:
            for line in child.stdout:
                print(line,end='',flush=True)
            if child.wait():
                raise RuntimeError('Remote process failed or reached its bounded deadline')
        finally:
            timer.cancel()

    try:
        archive = root/'payload.tar.gz'
        if not archive.exists():
            response = client.get('https://www.googleapis.com/drive/v3/files/'+settings['payload_file_id'],params={'alt':'media'},stream=True,timeout=120)
            response.raise_for_status()
            digest=hashlib.sha256()
            with archive.open('wb') as target:
                for block in response.iter_content(8*1024**2):
                    target.write(block);digest.update(block)
            assert digest.hexdigest()==settings['payload_sha256']
        code=root/'payload'
        code.mkdir(exist_ok=True)
        with tarfile.open(archive,'r:gz') as bundle:
            for member in bundle.getmembers():
                name=Path(member.name)
                if name.is_absolute() or '..' in name.parts or not(member.isfile() or member.isdir()):
                    raise ValueError('Unsafe archive member')
            bundle.extractall(code,filter='data')
        manifest=json.loads((code/'payload_manifest.json').read_text())
        for name,checksum in manifest['sha256'].items():
            digest=hashlib.sha256()
            with (code/name).open('rb') as source:
                for block in iter(lambda:source.read(8*1024**2),b''):digest.update(block)
            assert digest.hexdigest()==checksum, name
        jobs=code/'jobs/colab_continue'
        env=root/'venv'
        if not (env/'bin/python').exists():
            stream([sys.executable,'-m','pip','install','uv==0.8.15'],settings['deadline_unix']-1200)
            stream([sys.executable,'-m','uv','venv','--python','3.12','--seed',str(env)],settings['deadline_unix']-1200)
        python=str(env/'bin/python')
        stream([python,'-m','pip','install','torch==2.7.1','--index-url','https://download.pytorch.org/whl/cu128'],settings['deadline_unix']-1200)
        stream([python,'-m','pip','install','-r',str(jobs/'requirements.txt')],settings['deadline_unix']-1200)
        stream([python,'-u',str(jobs/'run_continue.py')],settings['deadline_unix']-900)
        result=json.loads((root/'training_result.json').read_text())
        # Separate process releases training memory before loading inference weights.
        evaluation_folder=settings['evaluation_folder_id']
        eval_jobs=code/'jobs/colab_recall'
        evaluator=(jobs/'evaluate_top200.py').read_text()
        eval_jobs.mkdir(exist_ok=True)
        (eval_jobs/'evaluate.py').write_text(evaluator)
        shutil.copy2(jobs/'drive_store.py',eval_jobs/'drive_store.py')
        remaining=max(0,settings['deadline_unix']-time.time()-300)
        if remaining<120:
            raise RuntimeError('Training artifacts saved; insufficient time for final full evaluation')
        stream([python,'-u',str(eval_jobs/'evaluate.py'),'--bundle',result['bundle'],'--inputs',str(code/'evaluation_inputs'),
            '--output',str(root/'evaluation'),'--credentials',str(credentials),'--folder-id',evaluation_folder,
            '--quota-project',settings['quota_project'],'--max-hours',str(min(3,remaining/3600)),
            '--training-complete-epochs',str(int(result['best_checkpoint_epoch_fraction'])),
            '--best-checkpoint-epoch-fraction',str(result['best_checkpoint_epoch_fraction'])],settings['deadline_unix']-240)
        evaluation=json.loads((root/'evaluation/status.json').read_text())
        result.update(status='completed' if evaluation['status']=='completed' else 'evaluation_partial',phase='completed',evaluation_status=evaluation['status'],evaluation_queries=evaluation['completed_queries'])
        (root/'result.json').write_text(json.dumps(result,indent=2))
        sys.path.insert(0,str(jobs))
        from drive_store import DriveStore
        DriveStore(credentials,settings['folder_id'],settings['quota_project']).put(root/'result.json','reports/job_status.json')
        print(json.dumps({'status':result['status'],'evaluation_queries':result['evaluation_queries']}),flush=True)
    finally:
        release()


if __name__=='__main__':
    main()
