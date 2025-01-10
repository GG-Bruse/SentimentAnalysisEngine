import os
import argparse
import json

def main(args):
    print(args.python_inference_path)
    print(args.engine_inference_path)
    output_classes = os.environ["output_classes"].split(',')
    output_numbers = int(os.environ["output_numbers"])
    with open(args.python_inference_path, 'r') as py_file, open(args.engine_inference_path, 'r') as en_file:
        python_inference_count = 0
        engine_inference_count = 0
        python_inference_list = []
        engine_inference_list = []

        for line in py_file:
            if python_inference_count == args.diff_count:
                break
            python_inference_count += 1

            data = json.loads(line)
            dict = {}
            for i in range(output_numbers):
                dict[output_classes[i]] = data[output_classes[i]]
            python_inference_list.append(dict)
        
        for line in en_file:
            if engine_inference_count == args.diff_count:
                break
            engine_inference_count += 1

            data = json.loads(line)
            dict = {}
            for i in range(output_numbers):
                dict[output_classes[i]] = data[output_classes[i]]
            engine_inference_list.append(dict)

        # print("python_inference_list", len(python_inference_list))
        # print("engine_inference_list", len(engine_inference_list))
        python = []
        engine = []
        total = 0
        for index in range(len(engine_inference_list)):
            py_row = []
            en_row = []
            for i in range(output_numbers):
                py_row.append('{:.5f}'.format(python_inference_list[index][output_classes[i]]))
                en_row.append('{:.5f}'.format(engine_inference_list[index][output_classes[i]]))
            python.append(py_row)
            engine.append(en_row)
            total += 1

        # 保留1到5位小数
        for decimal in range(1, 5):
            for cls in range(output_numbers):
                total_error = 0 
                total_diff = 0 
                for index in range(len(python)):
                    # print("this", python[index][cls])
                    # print("this2", python[index][cls][:decimal+2])
                    if python[index][cls][:decimal+2] != engine[index][cls][:decimal+2]:
                        total_diff += 1
                    diff_value = abs(float(python[index][cls]) - float(engine[index][cls]))
                    if diff_value > pow(0.1, decimal):
                        total_error += 1
                print('保留%s小数,在标签%s上,跳变率%.3f%%,差不匹配比例%.3f%%' %(decimal, output_classes[cls], total_diff / total * 100, total_error / total * 100))



            

            


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--python_inference_path', default='./model/python_inference_result.jsonl')
    argparser.add_argument('--engine_inference_path', default='./inference/engine_result_10000.jsonl')
    argparser.add_argument('--diff_count', default=10000)
    args = argparser.parse_args()
    main(args)