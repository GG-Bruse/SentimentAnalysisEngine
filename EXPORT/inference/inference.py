import tensorrt as trt
import argparse
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from tqdm import tqdm
from transformers import BertTokenizer

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class BertEngine(object):
    def __init__(self, engine_path, batch_size, max_seq_length):
        self.engine_path = engine_path
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.Time = 0

    def ready_inference(self):
        with open(self.engine_path, 'rb') as file, trt.Runtime(TRT_LOGGER) as runtime, \
            runtime.deserialize_cuda_engine(file.read()) as engine, engine.create_execution_context() as context:

            selected_profile = -1
            num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles # 4 // 1 = 4
            # print(engine.num_bindings)
            tensor_name = 'input_ids' # or 'segment_ids' or 'input_mask'
            for idx in range(engine.num_optimization_profiles):
                profile_shape = engine.get_tensor_profile_shape(name = tensor_name, profile_index = idx)
                print("profile shape:", profile_shape)
                if profile_shape[0][0] <= self.batch_size and profile_shape[2][0] >= self.batch_size and profile_shape[0][1] <= self.max_seq_length and profile_shape[2][1] >= self.max_seq_length:
                    selected_profile = idx
                    break
            if selected_profile == -1:
                raise RuntimeError("Could not find any profile that can run batch size {}.".format(args.batch_size))

            context.set_optimization_profile_async(selected_profile, cuda.Stream().handle)

            binding_idx_offset = selected_profile * num_binding_per_profile # 0
            input_shape = (self.batch_size, self.max_seq_length)
            input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
            for binding in range(3):
                context.set_binding_shape(binding_idx_offset + binding, input_shape)
            assert context.all_binding_shapes_specified

            self.stream = cuda.Stream()

            # Allocate device memory for inputs.
            print('Before initializing d_inputs')
            self.d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
            # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
            self.h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(binding_idx_offset + 3)), dtype=np.float32)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)

            def inference(feature):
                start_time = time.time()
                cuda.memcpy_htod_async(self.d_inputs[0], feature['input_ids'], self.stream)
                cuda.memcpy_htod_async(self.d_inputs[1], feature['segment_ids'], self.stream)
                cuda.memcpy_htod_async(self.d_inputs[2], feature['input_mask'], self.stream)
                context.execute_async_v2(bindings=[0 for i in range(binding_idx_offset)] + [int(d_inp) for d_inp in self.d_inputs] + [int(self.d_output)], stream_handle=self.stream.handle)
                cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
                self.stream.synchronize()

                end_time = time.time()
                self.Time = end_time - start_time

                return self.h_output
        return inference





def main(args):
    batch_size = 5
    max_seq_length = 512

    bert_engine = BertEngine(args.engine_path, batch_size, max_seq_length)
    inference_engine = bert_engine.ready_inference()

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file.rstrip('vocab.txt'))
    def text2feature(text):
        tokens = tokenizer.tokenize(text)  # 分词
        encode_dict = tokenizer.encode_plus(
            text=tokens,
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True,
            padding='max_length' if max_seq_length is not None else 'longest',
            return_token_type_ids=True,
            return_attention_mask=True
        )
        return encode_dict

    infer_count = 0
    batch_count = 0
    batch_dic = {}
    input_ids_list = []
    segment_ids_list = []
    input_mask_list = []
    with open(args.input_path, 'r') as input_file, open('./inference/engine_result_' + str(args.inference_count) + '.csv', 'w') as output_file:
        for line in tqdm(input_file):
            if infer_count == args.inference_count:
                break

            line = line.strip()
            content, label = line.split('\t')
            feature = text2feature(content)
            input_ids_list.append(feature['input_ids'])
            segment_ids_list.append(feature['attention_mask'])
            input_mask_list.append(feature['token_type_ids'])
            infer_count += 1
            batch_count += 1
            if batch_count % batch_size == 0 and batch_count != 0:
                batch_dic['input_ids'] = np.array(input_ids_list, dtype = np.int32)
                batch_dic['segment_ids'] = np.array(segment_ids_list, dtype = np.int32)
                batch_dic['input_mask'] = np.array(input_mask_list, dtype = np.int32)

                infer_start = time.time()
                result = inference_engine(batch_dic)
                infer_end = time.time()
                print("output:", result)

                batch_dic = {}
                input_ids_list = []
                segment_ids_list = []
                input_mask_list = []



            




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--engine_path', default='./output/model.engine')
    argparser.add_argument('--vocab_file', default='./model/vocab.txt')
    argparser.add_argument('--input_path', default='./inference/test.txt')
    argparser.add_argument('--inference_count', default=10000)
    args = argparser.parse_args()
    main(args)

    