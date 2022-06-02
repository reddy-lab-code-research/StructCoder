# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, model,beam_size=None,max_length=None,args=None):
        super(Seq2Seq, self).__init__()
        self.model=model
        self.args=args

        self.beam_size=beam_size
        self.max_length=max_length
        
        self.ast_type_emb = nn.Embedding(len(args.node_types), model.config.d_model)
        self.ast_depth_emb = nn.Parameter(torch.empty((1, 1, args.max_ast_depth, model.config.d_model)), requires_grad=True)
        nn.init.normal_(self.ast_depth_emb)
        
        self.dfg_bits = 16
        self.dfg_weight1 = nn.Linear(self.dfg_bits, 32, bias=False)
        self.dfg_weight2 = nn.Linear(self.dfg_bits, 32, bias=False)
        self.dfg_b1 = nn.Linear(self.dfg_bits, 1, bias=False)
        self.dfg_b2 = nn.Linear(self.dfg_bits, 1, bias=False)
        self.dfg_b3 = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        
        self.ast_path_bits = 128
        self.ast_path_head = nn.Linear(self.ast_path_bits, args.max_ast_depth*len(args.node_types), bias=False)
        
        self.ast_weight = nn.Linear(self.ast_path_bits, 32, bias=False)
        self.ast_b1 = nn.Linear(self.ast_path_bits, 1, bias=False)
        self.ast_b2 = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        
        self.eps=1e-10
        
        self.alpha1 = nn.Parameter(torch.tensor(-4.0, requires_grad=True))
        self.alpha2 = nn.Parameter(torch.tensor(-4.0, requires_grad=True))
        self.alpha1_clip = None
        self.alpha2_clip = None
        
        self.enable_dfg_op = True
        self.enable_ast_op = True
        self.enable_dfg_ip = True
        self.enable_ast_ip = True
        
    def set_alpha(self, alpha1, alpha2):
        self.alpha1 = nn.Parameter(torch.tensor(alpha1, requires_grad=True))
        self.alpha2 = nn.Parameter(torch.tensor(alpha2, requires_grad=True))
        
    def set_alpha_clip(self, alpha1_clip, alpha2_clip):
        self.alpha1_clip = alpha1_clip
        self.alpha2_clip = alpha2_clip
        
    def disable_dfg_op(self):
        self.enable_dfg_op = False
        
    def disable_ast_op(self):
        self.enable_ast_op = False
        
    def disable_dfg_ip(self):
        self.enable_dfg_ip = False
        
    def disable_ast_ip(self):
        self.enable_ast_ip = False
        
    def forward(self, source_ids=None, dfg_ids=None, lr_paths=None, attention_mask=None, sim_mat=None,\
                labels=None, decoder_attention_mask=None, target_dfg=None, target_ast=None, target_ast_sim=None):
        
        if dfg_ids is not None: # input is code
            input_ids = source_ids
            if self.enable_dfg_ip:
                input_ids = torch.cat((input_ids, dfg_ids), axis=-1)
            input_embeds = self.model.shared(input_ids)
            if self.enable_ast_ip:
                mask = (lr_paths>=0).int() # b, num_leaves, max_depth
                lr_paths = torch.clip(lr_paths, min=0)
                ast_emb = self.ast_type_emb(lr_paths) * self.ast_depth_emb # b, num_leaves, max_depth, 768
                ast_emb = ast_emb * mask[:,:,:,None]
                ast_emb = ast_emb.sum(axis=2)
                input_embeds = torch.cat((input_embeds, ast_emb), axis=1)
            
            mask_rel_pos = torch.ones_like(source_ids[0])
            if self.enable_dfg_ip:
                mask_rel_pos = torch.cat((mask_rel_pos, torch.zeros_like(dfg_ids[0])), axis=-1)
            if self.enable_ast_ip:
                mask_rel_pos = torch.cat((mask_rel_pos, torch.zeros_like(lr_paths[0,:,0])), axis=-1)
                
            if not(self.enable_ast_ip):
                L = input_ids.size()[1]
                attention_mask = attention_mask[:, :L, :L]
                sim_mat = sim_mat[:, :L, :L]
            elif not(self.enable_dfg_ip):
                s = source_ids.size()[1]
                e = s+dfg_ids.size()[1]
                attention_mask = torch.cat((attention_mask[:, :s, :], attention_mask[:, e:, :]), axis=1)       
                attention_mask = torch.cat((attention_mask[:, :, :s], attention_mask[:, :, e:]), axis=-1)  
                sim_mat = torch.cat((sim_mat[:, :s, :], sim_mat[:, e:, :]), axis=1)       
                sim_mat = torch.cat((sim_mat[:, :, :s], sim_mat[:, :, e:]), axis=-1)  
                                              
#             print (input_embeds.size(), '='*100)
#             print (attention_mask.size(), '='*100)
#             print (mask_rel_pos.size(), '='*100)

        if labels is not None:  
            
            if dfg_ids is not None: # input is code
                outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, \
                                 labels=labels, decoder_attention_mask=decoder_attention_mask, \
                                 mask_rel_pos=mask_rel_pos, output_hidden_states=True, sim_mat=sim_mat)
            else: # input is comment
                outputs = self.model(input_ids=source_ids, attention_mask=attention_mask, \
                                    labels=labels, decoder_attention_mask=decoder_attention_mask, \
                                    output_hidden_states=True)
            
            transl_loss = outputs.loss
            if target_dfg is None: # target is comment
                return transl_loss
            
            hidden = outputs.decoder_hidden_states[-1] # b, L, d
            
            
            if self.alpha1_clip is not None:
                alpha1 = torch.sigmoid(torch.clip(self.alpha1,max=self.alpha1_clip))
            else:
                alpha1 = torch.sigmoid(self.alpha1)
            if self.alpha2_clip is not None:
                alpha2 = torch.sigmoid(torch.clip(self.alpha2,max=self.alpha2_clip))
            else:
                alpha2 = torch.sigmoid(self.alpha2)
            
            # DFG
            if self.enable_dfg_op:
                dfg_hidden = hidden[:, :, :self.dfg_bits]
                hidden1 = self.dfg_weight1(dfg_hidden) # b,L,d'
                hidden2 = self.dfg_weight2(dfg_hidden) # b,L,d'
                dfg_pred = torch.sigmoid( torch.bmm(hidden1, hidden2.permute([0, 2, 1]).contiguous()) + self.dfg_b1(dfg_hidden) + \
                                         self.dfg_b2(dfg_hidden).permute(0,2,1) + self.dfg_b3 ) # b,L,L
                
                if type(target_dfg)!=int:
                    bp = 0.5
                    tp = ((dfg_pred>=bp)*(target_dfg==1)).int().sum()
                    tn = ((dfg_pred<bp)*(target_dfg==0)).int().sum()
                    dfg_pred = dfg_pred.reshape(-1, 1) #bLL
                    dfg_pred = torch.log(torch.hstack((1-dfg_pred+self.eps, dfg_pred+self.eps))) # 2, n

                    total = (target_dfg!=-1).long().sum()+2
                    num_pos = (target_dfg==1).long().sum()+1
                    loss_fct = nn.NLLLoss(ignore_index=-1, weight=torch.stack((total/(total-num_pos),total/num_pos)))
                    dfg_loss = loss_fct(dfg_pred, target_dfg.clone().view(-1))
                    target_dfg[:,0,:] = -1
                    target_dfg[:,:,0] = -1
                    dfg_ret = {'loss':dfg_loss, 'tp':tp, 'tn':tn, 'pos':num_pos, 'total':total, \
                          'pred':dfg_pred, 'true':target_dfg.view(-1)}
            else:
                dfg_loss, alpha1 = torch.tensor(0.0).to(self.args.device), 0
                dfg_ret = {'loss':dfg_loss, 'tp':torch.tensor(1).to(self.args.device), 'tn':torch.tensor(1).to(self.args.device),\
                           'pos':torch.tensor(2).to(self.args.device), 'total':torch.tensor(4).to(self.args.device)}

            # AST paths
            if self.enable_ast_op:
                ast_pred = self.ast_path_head(hidden[:, :, self.dfg_bits:self.dfg_bits+self.ast_path_bits]) # b,L,max_depth,num_node_types
                ast_pred = ast_pred.view(-1, ast_pred.size()[1], self.args.max_ast_depth, \
                                         len(self.args.node_types)) # b, L, max_depth, num_node_types
                if type(target_dfg)!=int:
                    ast_pred = ast_pred.permute([0, 3, 1, 2]).contiguous() # b, C, L, D

                    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                    ast_path_loss = loss_fct(ast_pred, target_ast)

                    pred = torch.argmax(ast_pred, dim=1) # b, L, D
                    match, total  = (pred==target_ast).sum(), (target_ast>=0).sum()
                    ast_path_ret = {'loss':ast_path_loss,'match':match,'total':total}
            else:
                alpha2, ast_path_loss = 0, torch.tensor(0.0).to(self.args.device)
                ast_path_ret = {'loss':ast_path_loss,'match':torch.tensor(1).to(self.args.device),\
                                'total':torch.tensor(1).to(self.args.device)}
                
            
            if type(target_dfg)==int:
                return dfg_pred, ast_pred
            
            loss = (3-alpha1-alpha2)*transl_loss + alpha1*dfg_loss + alpha2*ast_path_loss 
            
            return loss, transl_loss, dfg_ret, ast_path_ret
            
        else:
            if dfg_ids is not None: # input is code
                preds = self.model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, mask_rel_pos=mask_rel_pos,
                                        use_cache=True, num_beams=self.args.beam_size, early_stopping=False, 
                                        max_length=self.args.max_target_length, sim_mat=sim_mat)
            else: # input is comment
                preds = self.model.generate(input_ids=source_ids, attention_mask=attention_mask, \
                                    use_cache=True, num_beams=self.args.beam_size, early_stopping=False, 
                                        max_length=self.args.max_target_length)
            
            return preds
        
        
        
        
        