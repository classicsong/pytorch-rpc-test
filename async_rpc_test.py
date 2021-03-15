import torch as th
import torch.nn as nn
import argparse
import time

import torch.distributed.rpc as rpc

class DataServer(nn.Module):
    def __init__(self, data, i):
       super(DataServer, self).__init__()
       self.data = data

    def forward(self, idx):
        return self.data[idx]

class Server():
    def __init__(self, args, data):
        self.data = data
        self.run(args)

    def run(self, args):
        if args.rank == 0:
            self.run_client(args)
        else:
            self.run_server(args)

    def run_client(self, arg):
        name = arg.name
        remote = arg.remote
        rpc.init_rpc(name, rank=arg.rank, world_size=arg.world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.workers))
                     #rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(num_send_recv_threads=args.workers))
        rref = rpc.remote(remote, DataServer, args=(self.data, 1))
        randidx = th.randint(self.data.shape[0], (10000,))
        result = rref.rpc_sync().forward(randidx)

        randidx = th.randint(self.data.shape[0], (10000,))
        async_result = []
        t1 = time.time()
        for i in range(args.workers):
            async_result.append(rref.rpc_async().forward(randidx))
        th.futures.wait_all(async_result)
        t2 = time.time()
        print('RTT of {} tensor is {}'.format(result.shape, t2-t1))
        rpc.shutdown()

    def run_server(self, args):
        rpc.init_rpc(args.name, rank=args.rank, world_size=args.world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.workers))
                     #rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(num_send_recv_threads=args.workers))

        rpc.shutdown()


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--size', type=int, default=100 * 1024)
        self.add_argument('--world_size', type=int, default=2)
        self.add_argument('--rank', type=int, default=0)
        self.add_argument('--name', type=str)
        self.add_argument('--remote', type=str)
        self.add_argument('--workers', type=int, default=8)

if __name__ == '__main__':
    args = ArgParser().parse_args()

    data = th.rand((args.size, 10))
    server = Server(args, data)
    
