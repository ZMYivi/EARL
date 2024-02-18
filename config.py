import argparse

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

def get_options(parser=argparse.ArgumentParser()):  
    
    parser.add_argument('--is_train', type=int, default=1, help='Set to False to inference.')
    parser.add_argument('--symbols', type=int, default=30000, help='vocabulary size.')  
    parser.add_argument('--set_size', type=int, default=900, help='Size of each model layer.')  
    parser.add_argument('--max_length', type=int, default=60, help='Max length of response.')  
    parser.add_argument('--embed_units', type=int, default=300, help="Size of word embedding.")
    parser.add_argument('--units', type=int, default=512, help="Size of each model layer.")
    parser.add_argument('--triple_num', type=int, default=20, help="Num of triples.")
    parser.add_argument('--layers', type=int, default=2, help='Number of layers in the model.')
    parser.add_argument('--beam_size', type=int, default=20, help='Beam size to use during beam inference.')  
    parser.add_argument('--beam_use', type=int, default=0, help='use beam search or not.')  
    parser.add_argument('--mask_use', type=int, default=1, help='use masked kg or not.')  
    parser.add_argument('--mem_use', type=int, default=0, help="use memory or not.")
    parser.add_argument('--post_process', type=int, default=0, help="use post process or not.")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size to use during training.")
    parser.add_argument('--data_dir', type=str, default='./data/duconv', help='Data directory')  
    parser.add_argument('--train_dir', type=str, default='./train', help='Training directory.')  
    parser.add_argument('--per_checkpoint', type=int, default=1000, help='How many steps to do per checkpoint.')  
    parser.add_argument('--inference_version', type=int, default=0, help="The version for inferencing.")
    parser.add_argument('--log_parameters', type=int, default=1, help="Set to True to show the parameters")
    parser.add_argument('--inference_path', type=str, default="", help="Set filename of inference, default isscreen")
  
    opt = parser.parse_args()  
    return opt  
  
if __name__ == '__main__':  
    opt = get_options()