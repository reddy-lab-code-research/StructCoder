from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
import re
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import (BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
from datasets import load_dataset
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   tree_to_token_nodes,
                   index_to_code_token,
                   tree_to_variable_index, 
                   detokenize_code)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from tree_sitter import Language, Parser
sys.path.append('CodeBLEU')
from calc_code_bleu import calc_code_bleu, calc_code_bleu_multilang
keywords_dir = 'CodeBLEU/keywords'

MODEL_CLASSES = {'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


dfg_function={
    'java':DFG_java,
    'c_sharp':DFG_csharp
}

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
parsers['cs'] = parsers['c_sharp']
    
def extract_structure(code, parser, lang):  
    try:
        # ast
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        ast_token_nodes = tree_to_token_nodes(root_node)
        tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index] 
        
        # dfg
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg,ast_token_nodes

    

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(split, args):
    if (split=='valid') or (split=='dev'):
        split='validation'
    dataset = load_dataset('code_x_glue_cc_code_to_code_trans')[split]
    examples = []
    for eg in dataset:
        examples.append(Example(idx = eg['id'], source=eg[args.source_lang][:-1], target=eg[args.target_lang][:-1]))
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 dfg_to_code,
                 dfg_to_dfg,
                 lr_paths,
                 leaf_to_code,
                 leaf_to_leaf,
                 target_dfg,
                 target_ast,
                 target_ast_sim
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.lr_paths = lr_paths
        self.leaf_to_code = leaf_to_code
        self.leaf_to_leaf = leaf_to_leaf
        self.target_dfg = target_dfg
        self.target_ast = target_ast
        self.target_ast_sim = target_ast_sim

    
def get_node_types(node, l):
    l.append(node.type)
    for child in node.children:
        get_node_types(child, l)
        
        
def gather_node_types(examples, args):
    global node_types
    filename = args.output_dir+'/node_types.pkl'
    node_types = []
    for example in tqdm(examples):
        root = parsers[args.source_lang][0].parse(bytes(example.source,'utf8')).root_node 
        get_node_types(root, node_types)
        root = parsers[args.target_lang][0].parse(bytes(example.target,'utf8')).root_node 
        get_node_types(root, node_types)
    node_types = sorted(list(set(node_types)))
    pickle.dump(node_types, open(filename, 'wb'))
    node_types = {t:i for i,t in enumerate(node_types)}

def get_lr_path(leaf):
    if leaf==-1:
        return -1
    path = [leaf]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path

def convert_path_to_idx(path, max_depth):
    if path==-1:
        return [-1]*max_depth
    path = [node_types.get(node.type, -1) for node in path][:max_depth]
    path = path + [-1]*(max_depth-len(path))
    return path
        
def get_ll_sim(p1, p2): 
    if (p1==-1) or (p2==-1):
        return -1
    common = 1
    for i in range(2, min(len(p1), len(p2))+1):
        if p1[-i]==p2[-i]:
            common += 1
        else:
            break
    return common*common / (len(p1)*len(p2))

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    global source_sim_bps, target_sim_bps
    features = []
    match, nomatch = 1,1
    smatch, snomatch = 1,1
    bar = tqdm(enumerate(examples), total=len(examples))
    for example_index, example in bar:
        #source
        target_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        code_tokens,dfg,ast = extract_structure(example.source, parsers[args.source_lang], args.source_lang)
        for i in range(1, len(ast)):
            if (ast[i].start_point[0]<ast[i-1].start_point[0]) or \
                    ((ast[i].start_point[0]==ast[i-1].start_point[0]) and (ast[i].start_point[1]<ast[i-1].start_point[1])):
                raise Exception("Leaves not ordered by position in sequence.")
                    
        tcode = list(''.join(target_tokens).replace('Ġ', ' ').replace('ĉ', '\t'))
        scode = list(''.join(code_tokens))
        tcode_to_scode = []
        j = 0
        for i in range(len(tcode)):
            if j<len(scode):
                if tcode[i]==scode[j]:
                    tcode_to_scode.append(j)
                    j += 1
                    smatch += 1
                else:
                    tcode_to_scode.append(-1)
                    if (tcode[i]!=' '):
                        snomatch += 1
            else:
                tcode_to_scode.append(-1)
                if (tcode[i]!=' '):
                    snomatch += 1
            
        tcode_to_target = []
        for i in range(len(target_tokens)):
            tcode_to_target += [i]*len(target_tokens[i])
        scode_to_code = []
        for i in range(len(code_tokens)):
            scode_to_code += [i]*len(code_tokens[i])
        
        target_to_code = [[] for i in range(len(target_tokens))]
        for i in range(len(tcode)):
            if tcode_to_scode[i]>=0:
                target_to_code[tcode_to_target[i]].append( scode_to_code[tcode_to_scode[i]] )
                
        target_to_code = [set(v) for v in target_to_code]
        max_code_tokens = max([max(v) for v in target_to_code if len(v)>0]) + 1
                
        code_to_target = [[] for i in range(max_code_tokens)]
        for i in range(len(target_to_code)):
            for c in target_to_code[i]:
                code_to_target[c].append(i+1) # account for adding CLS at beginning
                
        dfg_small = []
        for t in dfg:
            if t[1]<max_code_tokens:
                rights = [i for i in t[4] if i<max_code_tokens]
                if len(rights)>0:
                    dfg_small.append((t[0],t[1],t[2],t[3],rights))
        dfg = dfg_small.copy()
        ast = ast[:max_code_tokens]
                
        source_tokens =[tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        
        dfg=dfg[:args.max_dfg_length]
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[code_to_target[x[1]] for x in dfg]
        
        ast = ast[:args.max_ast_length]
        lr_paths = [get_lr_path(leaf) for leaf in ast]
        leaf_to_leaf = np.ones((len(ast), len(ast)))
        for i in range(len(ast)):
            for j in range(i+1, len(ast)):
                sim = get_ll_sim(lr_paths[i], lr_paths[j])
                leaf_to_leaf[i, j] = sim
                leaf_to_leaf[j, i] = sim
        lr_paths = [convert_path_to_idx(path, args.max_ast_depth) for path in lr_paths]
        leaf_to_code = [code_to_target[i] for i in range(len(ast))]

        
        #target
        if stage=="test":
            target_ids = -1
            target_dfg = -1
            target_ast = -1
            target_ast_sim = -1
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
            code_tokens,dfg,ast = extract_structure(example.target, parsers[args.target_lang], args.target_lang)
            for i in range(1, len(ast)):
                if (ast[i].start_point[0]<ast[i-1].start_point[0]) or \
                        ((ast[i].start_point[0]==ast[i-1].start_point[0]) and (ast[i].start_point[1]<ast[i-1].start_point[1])):
                    raise Exception("Leaves not ordered by position in sequence.")
                    
            tcode = list(''.join(target_tokens).replace('Ġ', ' ').replace('ĉ', '\t'))
            scode = list(''.join(code_tokens))
            tcode_to_scode = []
            j = 0
            for i in range(len(tcode)):
                if j<len(scode):
                    if tcode[i]==scode[j]:
                        tcode_to_scode.append(j)
                        j += 1
                        match += 1
                    else:
                        tcode_to_scode.append(-1)
                        if (tcode[i]!=' '):
#                             logger.info(tcode[i])
                            nomatch += 1
                else:
                    tcode_to_scode.append(-1)
                    if (tcode[i]!=' '):
#                         logger.info(tcode[i])
                        nomatch += 1
            
            tcode_to_target = []
            for i in range(len(target_tokens)):
                tcode_to_target += [i]*len(target_tokens[i])
            scode_to_code = []
            for i in range(len(code_tokens)):
                scode_to_code += [i]*len(code_tokens[i])
                
            target_to_code = [[] for i in range(len(target_tokens))]
            for i in range(len(tcode)):
                if tcode_to_scode[i]>=0:
                    target_to_code[tcode_to_target[i]].append( scode_to_code[tcode_to_scode[i]] )
                    
            target_to_code = [set(v) for v in target_to_code]
            max_code_tokens = max([max(v) for v in target_to_code if len(v)>0]) + 1
                    
            code_to_target = [[] for i in range(max_code_tokens)]
            for i in range(len(target_to_code)):
                for c in target_to_code[i]:
                    code_to_target[c].append(i+1) # don't account for adding CLS at beginning
                    
            dfg_small = []
            for t in dfg:
                if t[1]<max_code_tokens:
                    rights = [i for i in t[4] if i<max_code_tokens]
                    if len(rights)>0:
                        dfg_small.append((t[0],t[1],t[2],t[3],rights))
            dfg = dfg_small.copy()
            ast = ast[:max_code_tokens]
                
            target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            
            target_len = len(target_ids)
            target_dfg = np.zeros((target_len, target_len))
            target_ast = -np.ones((target_len, args.max_ast_depth))
            target_ast_sim = -np.ones((target_len, target_len))
            
            tlr_paths = [get_lr_path(leaf) for leaf in ast]
            tleaf_to_leaf = np.ones((len(ast), len(ast)))
            for i in range(len(ast)):
                for j in range(i+1, len(ast)):
                    sim = get_ll_sim(tlr_paths[i], tlr_paths[j])
                    tleaf_to_leaf[i, j] = sim
                    tleaf_to_leaf[j, i] = sim
                    
            tlr_paths = [convert_path_to_idx(path, args.max_ast_depth) for path in tlr_paths]
            for i,ts in enumerate(code_to_target):
                target_ast[ts, :] = np.array(tlr_paths[i]).reshape((1,-1))
                for i2,ts2 in enumerate(code_to_target):
                    sim = tleaf_to_leaf[i,i2]
                    for ts_ in ts:
                        target_ast_sim[ts_, ts2] = sim
                        
            for _,l,_,_,rs in dfg:
                for lt in code_to_target[l]:
                    for r in rs:
                        target_dfg[lt, code_to_target[r]] = 1
            target_dfg[-1,:] = -1
            target_dfg[:,-1] = -1
        
            
        bar.set_description(str(stage)+' '+str(snomatch/(smatch+snomatch))+' '+str(nomatch/(match+nomatch)))
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 dfg_to_code,
                 dfg_to_dfg,
                 lr_paths,
                 leaf_to_code,
                 leaf_to_leaf,
                 target_dfg,
                 target_ast,
                 target_ast_sim
            )
        )
        
    if stage=='train':
        ssims = []
        tsims = []
        for eg in tqdm(features):
            ssims.append(eg.leaf_to_leaf.flatten())
            tsims.append(eg.target_ast_sim.flatten())
        ssims = np.concatenate(ssims)
        ssims = ssims[ssims!=-1]
        tsims = np.concatenate(tsims)
        tsims = tsims[tsims!=-1]
        source_sim_bps = [np.percentile(ssims, p) for p in range(0,100,20)] + [100]
        target_sim_bps = [np.percentile(tsims, p) for p in range(0,100,20)] + [100]
        
    return features

class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):
        return self.examples[item]

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default="codet5", type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str, 
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base",
                        help="Pretrained tokenizer name or path if not the same as model_name")    
    parser.add_argument("--load_model_path", default="saved_models/pretrain/checkpoint-12000/pytorch_model.bin", type=str, 
                        help="Path to trained model: Should contain the .bin files" )  
    parser.add_argument("--config_name", default="Salesforce/codet5-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    
    ## Other parameters
    parser.add_argument("--source_lang", default='java', type=str,
                        help="source language")  
    parser.add_argument("--target_lang", default='cs', type=str,
                        help="target language")  
    
    parser.add_argument("--max_source_length", default=320, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_dfg_length", default=75, type=int)
    parser.add_argument("--max_ast_length", default=250, type=int)
    parser.add_argument("--max_ast_depth", default=12, type=int)
    parser.add_argument("--max_target_length", default=320, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--alpha", default=None, type=str)
    parser.add_argument("--alpha1_clip", default=None, type=float)
    parser.add_argument("--alpha2_clip", default=None, type=float)
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=25, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=500, type=int,
                        help="")
    parser.add_argument("--train_steps", default=50000, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    args.output_dir = 'saved_models/translation/'+args.source_lang+'-'+args.target_lang+'_target_length'+str(args.max_target_length)+'/'
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, local_files_only=True)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=args.do_lower_case, local_files_only=True)
    
    # Get node types.
    train_examples = read_examples('train', args)
    gather_node_types(train_examples, args)
    args.node_types = node_types
    filename = args.output_dir + 'train_features.pkl'
    if os.path.exists(filename):
        train_features = pickle.load(open(filename, 'rb'))
    else:
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        pickle.dump(train_features, open(filename, 'wb'))
    
    #budild model
    model = model_class.from_pretrained(args.model_name_or_path, local_files_only=True)
    model = Seq2Seq(model=model, beam_size=args.beam_size,max_length=args.max_target_length, args=args)
    
    if args.load_model_path!='none':
        logger.info("reload model from {}".format(args.load_model_path))
        pt_dict = torch.load(args.load_model_path)
        my_dict = model.state_dict()
        for k,v in pt_dict.items():
            if k not in ['ast_type_emb.weight', 'ast_path_head.weight', 'ast_path_head.bias']:
                my_dict[k] = v
        pt_node_types = pickle.load(open('pretrain/pt_node_types_3L_cls_target.pkl','rb'))
        pt_node_types = {k:i for i,k in enumerate(pt_node_types)}
        my_to_pt_node_types = [-1 for i in range(len(node_types))]
        for k,i in node_types.items():
            if k in pt_node_types:
                my_to_pt_node_types[i] = pt_node_types[k]
                with torch.no_grad():
                    my_dict['ast_type_emb.weight'][i,:] = pt_dict['ast_type_emb.weight'][pt_node_types[k], :]
                    my_dict['ast_path_head.weight'][i::len(node_types), :] = pt_dict['ast_path_head.weight'][i::len(pt_node_types), :]
        logger.info("*********** No. of new node types = %d", (np.array(my_to_pt_node_types)==-1).sum())
        model.load_state_dict(my_dict)
        if args.alpha1_clip is not None:
            model.set_alpha_clip(args.alpha1_clip, args.alpha2_clip)
        
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

        
    def collate_batch(batch):
        source_ids_list, target_ids_list = [], []
        dfg_to_code_list, dfg_to_dfg_list = [], []
        leaf_to_code_list, leaf_to_leaf_list, lr_paths_list = [], [], []
        target_dfg_list, target_ast_list, target_ast_sim_list = [], [], []
        for eg in batch:
            source_ids_list.append(eg.source_ids)
            target_ids_list.append(eg.target_ids)
            dfg_to_code_list.append(eg.dfg_to_code)
            dfg_to_dfg_list.append(eg.dfg_to_dfg)
            leaf_to_code_list.append(eg.leaf_to_code)
            leaf_to_leaf_list.append(eg.leaf_to_leaf)
            lr_paths_list.append(eg.lr_paths)
            target_dfg_list.append(eg.target_dfg)
            target_ast_list.append(eg.target_ast)
            target_ast_sim_list.append(eg.target_ast_sim)
            
        not_test = (target_ids_list[0]!=-1)
        max_source_len = max([len(l) for l in source_ids_list])
        max_dfg_len = max([len(l) for l in dfg_to_code_list])
        max_ast_len = max([len(l) for l in leaf_to_code_list])
        if not_test:
            max_target_len = max([len(l) for l in target_ids_list])
            
        batch_size = len(source_ids_list)
        
        attention_mask = np.zeros((batch_size, max_source_len+max_dfg_len+max_ast_len, max_source_len+max_dfg_len+max_ast_len))
        sim_mat = -np.ones((batch_size, max_source_len+max_dfg_len+max_ast_len, max_source_len+max_dfg_len+max_ast_len))
        lr_paths = -np.ones((batch_size, max_ast_len, args.max_ast_depth))
        dfg_ids = tokenizer.pad_token_id*np.ones((batch_size, max_dfg_len))
        if not_test:
            target_mask = np.zeros((batch_size, max_target_len))
            target_dfg = -np.ones((batch_size, max_target_len, max_target_len))
            target_ast = -np.ones((batch_size, max_target_len, args.max_ast_depth))
            target_ast_sim = -np.ones((batch_size, max_target_len, max_target_len))
            
        for i in range(batch_size):
            curr_len = len(source_ids_list[i])
            pad_len = max_source_len - curr_len
            source_ids_list[i] = source_ids_list[i] + [tokenizer.pad_token_id]*pad_len
            attention_mask[i, :curr_len, :curr_len] = 1
            
            curr_dfg_len = len(dfg_to_code_list[i])
            dfg_ids[i, :curr_dfg_len] = tokenizer.unk_token_id
            for ind,comesfrom in enumerate(dfg_to_dfg_list[i]):
                if len(comesfrom)>0:
                    attention_mask[i, max_source_len+ind, max_source_len+np.array(comesfrom)] = 1 
            for ind,tokens in enumerate(dfg_to_code_list[i]):
                attention_mask[i, max_source_len+ind, tokens] = 1 
                attention_mask[i, tokens, max_source_len+ind] = 1 
                
            attention_mask[i, 0, max_source_len:max_source_len+curr_dfg_len] = 1
            attention_mask[i, max_source_len:max_source_len+curr_dfg_len, 0] = 1
            attention_mask[i, max_source_len:max_source_len+curr_dfg_len, curr_len-1] = 1
            attention_mask[i, curr_len-1, max_source_len:max_source_len+curr_dfg_len] = 1
            
            curr_ast_len = len(leaf_to_code_list[i])
            lr_paths[i, :curr_ast_len, :] = np.array(lr_paths_list[i])
            attention_mask[i, max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len, \
                          max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len] = 1 
            for ind,tokens in enumerate(leaf_to_code_list[i]):
                attention_mask[i, max_source_len+max_dfg_len+ind, tokens] = 1 
                attention_mask[i, tokens, max_source_len+max_dfg_len+ind] = 1 
                
            attention_mask[i, 0, max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len] = 1
            attention_mask[i, max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len, 0] = 1
            attention_mask[i, max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len, curr_len-1] = 1
            attention_mask[i, curr_len-1, max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len] = 1
            
            sim_mat[i, max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len, \
                    max_source_len+max_dfg_len:max_source_len+max_dfg_len+curr_ast_len] = leaf_to_leaf_list[i]
            
            
            if not_test:
                curr_len = len(target_ids_list[i])
                pad_len = max_target_len - curr_len
                target_ids_list[i] = target_ids_list[i] + [-100]*pad_len
                target_mask[i,:curr_len] = 1
                target_dfg[i, :curr_len, :curr_len] = target_dfg_list[i]
                target_ast[i, :curr_len, :] = target_ast_list[i]
                target_ast_sim[i, :curr_len, :curr_len] = target_ast_sim_list[i]
                
        mask = (sim_mat==-1).astype(int)
        sim_mat = -10000*mask + (1-mask)*np.log(np.clip(sim_mat, a_min=1e-10, a_max=np.inf))
                
        if not_test:
            return torch.tensor(source_ids_list).long(), \
                   torch.tensor(dfg_ids).long(), \
                   torch.tensor(lr_paths).long(), \
                   torch.tensor(attention_mask).int(), \
                   torch.tensor(sim_mat).float(), \
                   torch.tensor(target_ids_list), \
                   torch.tensor(target_mask).int(), \
                   torch.tensor(target_dfg).long(), \
                   torch.tensor(target_ast).long(), \
                   torch.tensor(target_ast_sim).float(),
            
        return torch.tensor(source_ids_list).long(), \
               torch.tensor(dfg_ids).long(), \
               torch.tensor(lr_paths).long(), \
               torch.tensor(attention_mask).int(), \
               torch.tensor(sim_mat).float()
            
# node_ids, attention_mask            
            
            
    if args.do_train:
        # Prepare training data loader
        train_data = TextDataset(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                      batch_size=args.train_batch_size//args.gradient_accumulation_steps,\
                                      collate_fn=collate_batch)
        

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
    
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps*args.train_batch_size//len(train_examples))
        

        model.train()
        dev_dataset={}
        
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_codebleu,best_loss = 0, 0,0,0,0,0,1e6 
        tr_transl_loss = 0
        tr_dfg_ret = {'loss':0, 'tp':0, 'tn':0, 'pos':0, 'total':0}
        tr_ast_path_ret = {'loss':0, 'match':0, 'total':0}
#         tr_ast_sim_ret = {'loss':0, 'match':0, 'total':0}
        
        bar = range(num_train_optimization_steps)
        train_dataloader=iter(train_dataloader)
        eval_flag = True
        for step in bar:
            try:
                batch = next(train_dataloader)
            except StopIteration:
                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                      batch_size=args.train_batch_size//args.gradient_accumulation_steps,\
                                      collate_fn=collate_batch)
                train_dataloader=iter(train_dataloader)
                batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids,dfg_ids,lr_paths,attention_mask,sim_mat,target_ids,target_mask,target_dfg,target_ast,target_ast_sim = batch
            
            loss, transl_loss, dfg_ret, ast_path_ret = model(source_ids=source_ids, dfg_ids=dfg_ids, lr_paths=lr_paths, \
                                                      attention_mask=attention_mask, sim_mat=sim_mat, labels=target_ids, \
                                                     decoder_attention_mask=target_mask, target_dfg=target_dfg, target_ast=target_ast, \
                                                            target_ast_sim=target_ast_sim)
                        
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
                transl_loss = transl_loss.mean()
                dfg_ret['loss'] = dfg_ret['loss'].mean()
                ast_path_ret['loss'] = ast_path_ret['loss'].mean()
#                 ast_sim_ret['loss'] = ast_sim_ret['loss'].mean()
                for k in ['tp','tn','pos','total']:
                    dfg_ret[k] = dfg_ret[k].sum()
                for k in ['match','total']:
                    ast_path_ret[k] = ast_path_ret[k].sum()
#                     ast_sim_ret[k] = ast_path_ret[k].sum()
                    
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            tr_loss += loss.item()
            tr_transl_loss += transl_loss.item()
            tr_dfg_ret['loss'] += dfg_ret['loss'].item()
            tr_ast_path_ret['loss'] += ast_path_ret['loss'].item()
#             tr_ast_sim_ret['loss'] += ast_path_ret['loss'].item()
            for k in ['tp','tn','pos','total']:
                tr_dfg_ret[k] += dfg_ret[k].item()
            for k in ['match','total']:
                tr_ast_path_ret[k] += ast_path_ret[k].item()
#                 tr_ast_sim_ret[k] += ast_path_ret[k].item()
                
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True
                
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),3)
            train_transl_loss=round(tr_transl_loss/(nb_tr_steps+1),3)
#             train_ast_sim_loss=round(tr_ast_sim_ret['loss']/(nb_tr_steps+1),3)
            train_dfg_ret_loss = round(tr_dfg_ret['loss']/(nb_tr_steps+1),3)
            train_ast_path_loss = round(tr_ast_path_ret['loss']/(nb_tr_steps+1),3)
#             train_alpha = [round(x,5) for x in alpha/(nb_tr_steps+1)]
            
            
            if (global_step + 1)%100==0:
                logger.info("\n")
                logger.info(" step {} |loss {} |tra {} |dfg {} {} {} |ast-p {} {}".format(\
                                       global_step + 1,train_loss,train_transl_loss,\
                                       train_dfg_ret_loss, round(tr_dfg_ret['tp']/tr_dfg_ret['pos'],3), \
                                       round(tr_dfg_ret['tn']/(tr_dfg_ret['total']-tr_dfg_ret['pos']),3), \
                                       train_ast_path_loss, round(tr_ast_path_ret['match']/tr_ast_path_ret['total'],3) ))
#                 logger.info(" loss weights {} {} {}".format(train_alpha[0], train_alpha[1], train_alpha[2]))
                with torch.no_grad():
                    alpha1, alpha2 = model.module.alpha1.detach().cpu().numpy(), \
                                             model.module.alpha2.detach().cpu().numpy()
                logger.info(" loss weights {} {}".format(alpha1, alpha2))
                
            if args.do_eval and ((global_step + 1) %args.eval_steps == 0) and eval_flag:
                #Eval model with dev dataset
                nb_tr_examples, nb_tr_steps,tr_loss = 0, 0,0
                tr_transl_loss = 0
                tr_dfg_ret = {'loss':0, 'tp':0, 'tn':0, 'pos':0, 'total':0}
                tr_ast_path_ret = {'loss':0, 'match':0, 'total':0}
#                 tr_ast_sim_ret = {'loss':0, 'match':0, 'total':0}

                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples('dev', args)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    eval_data = TextDataset(eval_features)
                    dev_dataset['dev_loss']=eval_examples,eval_data
                    
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_batch)
                
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                batch_num = 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,dfg_ids,lr_paths,attention_mask,sim_mat,target_ids,target_mask,target_dfg,target_ast, target_ast_sim = batch
                    
                    with torch.no_grad():
                        loss, transl_loss, dfg_ret, ast_path_ret = model(source_ids=source_ids, dfg_ids=dfg_ids, \
                                                                    lr_paths=lr_paths, attention_mask=attention_mask, sim_mat=sim_mat, \
                                    labels=target_ids, decoder_attention_mask=target_mask,target_dfg=target_dfg,target_ast=target_ast, \
                                                                        target_ast_sim=target_ast_sim)
                            
                    eval_loss += loss.mean().item()
                    batch_num += 1
                    
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / batch_num
             
                
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                
#                 #save last checkpoint
#                 last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
#                 if not os.path.exists(last_output_dir):
#                     os.makedirs(last_output_dir)
#                 model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
#                 output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
#                 torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
#                     # Save best checkpoint for best ppl
#                     output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
#                     if not os.path.exists(output_dir):
#                         os.makedirs(output_dir)
#                     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
#                     output_model_file = os.path.join(output_dir, "pytorch_model.bin")
#                     torch.save(model_to_save.state_dict(), output_model_file)  
                            
                            
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples('dev', args)
#                     eval_examples = random.sample(eval_examples,min(500,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    eval_data = TextDataset(eval_features)
                    dev_dataset['dev_bleu']=eval_examples,eval_data
                    
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_batch)
               

                model.eval() 
                p=[]
                pred_ids = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)

                    source_ids,dfg_ids,lr_paths,attention_mask,sim_mat= batch                  
                    with torch.no_grad():
                        module = model.module if hasattr(model, 'module') else model
                        preds = module(source_ids, dfg_ids, lr_paths=lr_paths, attention_mask=attention_mask, sim_mat=sim_mat)
                        top_preds = list(preds.cpu().numpy())
                        pred_ids.extend(top_preds)
                        
                    p = [tokenizer.decode(id, skip_special_tokens=True, 
                                                     clean_up_tokenization_spaces=False)
                                              for id in pred_ids]
                model.train()
                accs=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
#                         ref = ref.replace('  NEW_LINE', ' NEW_LINE')
                        f.write(ref.replace('\n', ' ')+'\n')
                        f1.write(gold.target+'\n')     
                        accs.append(ref==gold.target)

                dev_bleu=round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")),2)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
                logger.info("  "+"*"*20)    
                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        module = model.module if hasattr(model, 'module') else model
        module.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-best-bleu/pytorch_model.bin')))
        files=['test']
        for idx,split in enumerate(files):   
            logger.info("Test split: {}".format(split))
            eval_examples = read_examples(split, args)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            eval_data = TextDataset(eval_features)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_batch)

            model.eval() 
            p=[]
            pred_ids = []
            for batch in tqdm(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                
                source_ids,dfg_ids,lr_paths,attention_mask, sim_mat = batch                  
                with torch.no_grad():
                    module = model.module if hasattr(model, 'module') else model
                    preds = module(source_ids, dfg_ids, lr_paths=lr_paths, attention_mask=attention_mask, sim_mat=sim_mat)
                    top_preds = list(preds.cpu().numpy())
                    pred_ids.extend(top_preds)

            p = [tokenizer.decode(id, skip_special_tokens=True, 
                                                 clean_up_tokenization_spaces=False)
                                          for id in pred_ids]
            model.train()
            accs=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):
#                     ref = ref.replace('  NEW_LINE', ' NEW_LINE')
                    f.write(ref.replace('\n', ' ')+'\n')
                    f1.write(gold.target+'\n')    
                    accs.append(ref==gold.target)
            dev_bleu=round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(split), 
                                 os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(split)),2)
            if args.target_lang=='cs':
                args.target_lang = 'c_sharp'
            dev_codebleu = [round(i*100, 2) for i in \
                            calc_code_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(split), \
                            os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(split), args.target_lang, keywords_dir)]
            
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  %s = %s "%("codebleu-4",str(dev_codebleu)))
            logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
            logger.info("  "+"*"*20)    
            


                            

                
                
if __name__ == "__main__":
    main()


