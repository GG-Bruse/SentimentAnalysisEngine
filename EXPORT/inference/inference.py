import tensorrt as trt
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class BertEngine(object):
    def __init__(self, engine_path, batch_size, max_seq_length):
        self.engine_path = engine_path
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def construct(self):
        with open(self.engine_path, 'rb') as file, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(file.read())


def main(args):
    batch_size = 5
    max_seq_length=512
    bert_engine = BertEngine(args.engine_path, batch_size, max_seq_length)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--engine_path', default='./output/model.engine')
    argparser.add_argument('--vocab_file', default='./model/vocab.txt')
    argparser.add_argument('--input_path', default='./model/vocab.txt')
    argparser.add_argument('--inference_count', default=10000)
    args = argparser.parse_args()
    main(args)

    