import torch as th
import argparse
import time

import torch.distributed.rpc as rpc

@th.jit.script
def collect(data: th.Tensor, input : int) -> th.Tensor:
    return data

def run(arg, data):
    name = arg.name
    remote = arg.remote
    rpc.init_rpc(name, rank=arg.rank, world_size=arg.world_size)
    result = rpc.rpc_sync(remote, collect, args=(data, 2))
    t1 = time.time()
    result = rpc.rpc_sync(remote, collect, args=(data, 2))
    t2 = time.time()
    print('RTT of {} tensor is {}'.format(result.shape, t2-t1))
    rpc.shutdown()

def start(args):
    rpc.init_rpc(args.name, rank=args.rank, world_size=args.world_size, num_worker_threads=8)

    rpc.shutdown()


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--size', type=int, default=1024 * 1024)
        self.add_argument('--world_size', type=int, default=2)
        self.add_argument('--rank', type=int, default=0)
        self.add_argument('--name', type=str)
        self.add_argument('--remote', type=str)
        self.add_argument('--async', type=bool)

if __name__ == '__main__':
    args = ArgParser().parse_args()

    data = th.empty((args.size, 128))
    if args.rank == 0:
        run(args, data)
    else:
        start(args)
