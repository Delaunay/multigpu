import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--devices', type=str, default=None)
    parser.add_argument('script', type=str)
    parser.add_argument('args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    processes = []

    args.devices = args.devices.split(',')
    print(args)

    job_env = os.environ.copy()
    for rank, device_id in enumerate(args.devices):
        cmd = [f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, "-u"]
        cmd.append(args.script)
        cmd.append('--distributed_dataparallel')
        cmd.extend(('--rank',  str(rank)))
        cmd.extend(('--world-size',  str(len(args.devices))))
        cmd.extend(('--dist-backend', 'nccl'))
        cmd.extend(('--dist-url',  'tcp://localhost:8181'))
        cmd.extend(args.args)
        
        print(cmd)
        process = subprocess.Popen(' '.join(cmd), env=job_env, shell=True)
        processes.append(process)

    errors = []

    for process in processes:
        process.wait()

        if process.returncode != 0:
            errors.append((process.returncode, cmd))

    for error in errors:
        print(error)


main()

