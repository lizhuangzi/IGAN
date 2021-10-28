from importlib import import_module
from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100','Manga109','DIV2KV','SunHay80','Testvideo','TestVideoPE']:
            if not args.benchmark_noise:
                if int(args.scale[0]) == 4:
                    module_test = import_module('data.benchmark4x')
                else:
                    module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=not args.cpu
        )