"""
This file contains mostly original code, with partial third-party snippets:
1. Classes 'SelfAttention', 'MultiHeadSelfAttention' and 'TransformerEncoderLayer' are modified from PyTorch (BSD 3-Clause License):
   - Source: https://github.com/pytorch
   - Copyright: Facebook, Inc. and its affiliates.
2. All other code is original and owned by Yiran Zhou.
"""

import numpy as np
import math
import json
import torch as th
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import esm


def genDataDict(l_pep_id,d_raw_predefined_appendix,d_raw_extended_appendix,dict_pep2pfam,dict_esm,dict_extended_appendix,token_prefix="[PRE]",token_null="[NULL]"):

    l_token_pfam_id=[]
    embedding_raw_predefined_appendix=[]
    embedding_raw_extended_appendix=[]
    l_prefix_index=[]
    l_pfam_index=[]
    l_null_pfam_index=[]
    l_non_null_pfam_index=[]
    dict_prefix_index2pfam_index_block={}
    dict_pfam_index2prefix_index={}
    dict_pfam_index2pfam_index_block={}
    dict_pfam_index2info={}
    embedding_esm=[]
    current_token_index=0
    for pep_id in l_pep_id:
        l_token_pfam_id.append(token_prefix)
        embedding_raw_predefined_appendix.append([0]*d_raw_predefined_appendix)
        embedding_raw_extended_appendix.append([0]*d_raw_extended_appendix)
        current_prefix_index=current_token_index
        current_token_index+=1
        l_pfam=[(pfam_id,start,end,relative_start,relative_end) for pfam_id,start,end,relative_start,relative_end in dict_pep2pfam[pep_id] if pfam_id in dict_extended_appendix]
        n_pfam=len(l_pfam)
        if n_pfam==0:
            l_token_pfam_id.append(token_null)
            embedding_raw_predefined_appendix.append([0]*d_raw_predefined_appendix)
            embedding_raw_extended_appendix.append([0]*d_raw_extended_appendix)
            pfam_index_block=[current_token_index]
            l_pfam_index.append(current_token_index)
            l_null_pfam_index.append(current_token_index)
            dict_pfam_index2info[current_token_index]=(0,0,0,0)
            current_token_index+=1
        else:
            pfam_index_block=[]
            for pfam_id,start,end,relative_start,relative_end in l_pfam:
                l_token_pfam_id.append(pfam_id)
                embedding_raw_predefined_appendix.append([relative_start,relative_end])
                embedding_raw_extended_appendix.append(dict_extended_appendix[pfam_id])
                pfam_index_block.append(current_token_index)
                l_pfam_index.append(current_token_index)
                l_non_null_pfam_index.append(current_token_index)
                dict_pfam_index2info[current_token_index]=(start,end,relative_start,relative_end)
                current_token_index+=1
        l_prefix_index.append(current_prefix_index)
        dict_prefix_index2pfam_index_block[current_prefix_index]=pfam_index_block
        for pfam_index in pfam_index_block:
            dict_pfam_index2prefix_index[pfam_index]=current_prefix_index
            dict_pfam_index2pfam_index_block[pfam_index]=pfam_index_block
        embedding_esm.append(dict_esm[pep_id])
        
    n_pep=len(l_pep_id)
    n_pfam=len(l_non_null_pfam_index)
    n_token=len(l_token_pfam_id)

    dict_data={
        "l_pep_id":l_pep_id,
        "n_pep":n_pep,"n_pfam":n_pfam,"n_token":n_token,
        "token_pfam_id":l_token_pfam_id,"embedding_raw_predefined_appendix":embedding_raw_predefined_appendix,"embedding_raw_extended_appendix":embedding_raw_extended_appendix,
        "prefix_index":l_prefix_index,"pfam_index":l_pfam_index,"non_null_pfam_index":l_non_null_pfam_index,"null_pfam_index":l_null_pfam_index,
        "dict_prefix_index2pfam_index_block":dict_prefix_index2pfam_index_block,"dict_pfam_index2prefix_index":dict_pfam_index2prefix_index,"dict_pfam_index2pfam_index_block":dict_pfam_index2pfam_index_block,"dict_pfam_index2info":dict_pfam_index2info,
        "embedding_esm":embedding_esm,
        "n_padding_pep":0,"n_padding_token":0,
        "max_n_pep_within_batch":n_pep,"max_n_pfam_within_batch":n_pfam,"max_n_token_within_batch":n_token
    }
    return dict_data
        

class MLP(nn.Module):
    def __init__(self,in_size,out_size,n_layer,activation_func,dropout_rate,input_dropout_off,output_activation_off):
        super().__init__()
        if n_layer<1:
            raise Exception()
        
        self.n_layer=n_layer
        l_layer_size=self.size_transition(in_size,out_size,self.n_layer)
        self.l_linear_layer=nn.ModuleList()
        for i in range(self.n_layer):
            self.l_linear_layer.append(nn.Linear(l_layer_size[i],l_layer_size[i+1]))
            
        self.activation_func=activation_func
        self.dropout=nn.Dropout(dropout_rate)
        self.input_dropout_off=input_dropout_off
        self.output_activation_off=output_activation_off
    
    def size_transition(self,in_size,out_size,n_layer):
        step=np.power(out_size/in_size,1/n_layer)
        l_hidden_layer_size=[]
        tmp_size=in_size
        for i in range(n_layer-1):
            tmp_size=tmp_size*step
            l_hidden_layer_size.append(int(tmp_size))
        return [in_size]+l_hidden_layer_size+[out_size]
    
    def forward(self,x):
        for i in range(self.n_layer):
            linear_layer=self.l_linear_layer[i]
            if self.input_dropout_off:
                if i!=0:
                    x=self.dropout(x)
                else:
                    pass
            else:
                x=self.dropout(x)
            x=linear_layer(x)
            if self.output_activation_off:
                if i!=self.n_layer-1:
                    x=self.activation_func(x)
                else:
                    pass
            else:
                x=self.activation_func(x)           
        return x
        

class SelfAttention(nn.Module):
    def __init__(self,d_model,n_head,dropout_rate):
        super(SelfAttention, self).__init__()
        self.d_model=d_model
        self.Wq=nn.Linear(d_model,d_model)
        self.Wk=nn.Linear(d_model,d_model)
        self.Wv=nn.Linear(d_model,d_model)
        self.multihead_attention_layer=nn.MultiheadAttention(embed_dim=d_model,num_heads=n_head,dropout=dropout_rate,batch_first=True)
        
    def forward(self,x,mask=None):
        q,k,v=self.Wq(x),self.Wk(x),self.Wv(x)
        attn_output,attn_output_weights=self.multihead_attention_layer(query=q,key=k,value=v,attn_mask=mask)
        return attn_output,attn_output_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0):
        super().__init__()
        
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.dropout=dropout
        self.head_dim=embed_dim//num_heads
        assert self.head_dim*num_heads==self.embed_dim,"embed_dim must be divisible by num_heads"
        
        self.Wq=nn.Linear(embed_dim,embed_dim)
        self.Wk=nn.Linear(embed_dim,embed_dim)
        self.Wv=nn.Linear(embed_dim,embed_dim)
        self.out_proj=nn.Linear(embed_dim,embed_dim)

    def forward(self,src,attn_mask=None,need_weight=False):
        if src.dim()!=3:
            raise Exception()
        bsz,tgt_len,embed_dim=src.shape
        if embed_dim!=self.embed_dim:
            raise Exception()
        src_len=tgt_len
        
        num_heads=self.num_heads
        head_dim=self.head_dim
        if attn_mask is not None:
            if attn_mask.shape!=(bsz,num_heads,tgt_len,tgt_len):
                raise Exception()
            attn_mask=F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=src.dtype,
                check_other=False,
            )

        if not self.training:
            dropout=0.0
        else:
            dropout=self.dropout
        
        q=self.Wq(src)
        k=self.Wk(src)
        v=self.Wv(src)

        q=q.view(bsz,tgt_len,num_heads,head_dim)
        k=k.view(bsz,src_len,num_heads,head_dim)
        v=v.view(bsz,src_len,num_heads,head_dim)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        if need_weight:
            q=q.view(bsz*num_heads,tgt_len,head_dim)
            k=k.view(bsz*num_heads,tgt_len,head_dim)
            v=v.view(bsz*num_heads,tgt_len,head_dim)
            B,Nt,E=q.shape
            q_scaled=q*math.sqrt(1.0/float(E))
            if attn_mask is not None:
                attn_mask=attn_mask.view(bsz*num_heads,tgt_len,tgt_len).to(q.dtype)
                attn_weight=th.baddbmm(attn_mask,q_scaled,k.transpose(-2,-1))
            else:
                attn_weight=th.bmm(q_scaled,k.transpose(-2,-1))
            attn_weight=F.softmax(attn_weight,dim=-1)
            if dropout>0.0:
                attn_weight=F.dropout(attn_weight,p=dropout)
            attn_output=th.bmm(attn_weight,v)
            attn_weight=attn_weight.view(bsz,num_heads,tgt_len,tgt_len)
            attn_output=attn_output.view(bsz,num_heads,tgt_len,head_dim).transpose(2,1).contiguous().view(bsz,tgt_len,embed_dim)
            attn_output=self.out_proj(attn_output)
            return attn_output,attn_weight
        else:
            attn_output=F.scaled_dot_product_attention(q,k,v,attn_mask,dropout)
            attn_output=attn_output.transpose(2,1).contiguous().view(bsz,tgt_len,embed_dim)
            attn_output=self.out_proj(attn_output)   
            return attn_output,None
        

class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,nhead,dim_feedforward,activation,dropout=0.1,layer_norm_eps=1e-5):
        super().__init__()
        
        self.self_attn=MultiHeadSelfAttention(d_model,nhead,dropout=dropout)
        self.linear1=nn.Linear(d_model,dim_feedforward)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(dim_feedforward,d_model)

        self.norm1=nn.LayerNorm(d_model,eps=layer_norm_eps)
        self.norm2=nn.LayerNorm(d_model,eps=layer_norm_eps)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

        self.activation=activation

    def forward(self,src,attn_mask=None,need_weight=False):
        x=src
        attn_output,attn_weight=self.self_attn(x,attn_mask,need_weight)
        attn_output=self.dropout1(attn_output)
        x=self.norm1(x+attn_output)
        ffn_output=self.linear2(self.dropout(self.activation(self.linear1(x))))
        ffn_output=self.dropout2(ffn_output)
        x=self.norm2(x+ffn_output)
        return x,attn_weight



class ESMModel(nn.Module):
    def __init__(self,d_encoding,pretrained_model_name):
        super().__init__()
        self.register_buffer("flag",th.tensor(True))
        self.model_esm,self.alphabet=eval("esm.pretrained."+pretrained_model_name+"()")
        self.linear_representation_projection=nn.Linear(self.model_esm.embed_dim,d_encoding)
        self.norm_representation_projection=nn.LayerNorm(d_encoding)
        self.linear_readout=nn.Linear(d_encoding,self.model_esm.alphabet_size)
        self.loss_func=nn.CrossEntropyLoss()
    
    def forward(self,l_pep_seq):
        device=self.flag.device
        
        data=[(i,pep_seq) for i,pep_seq in enumerate(l_pep_seq)]
        batch_converter=self.alphabet.get_batch_converter()
        batch_labels,batch_strs,batch_tokens=batch_converter(data)
        batch_tokens=batch_tokens.to(device)
        batch_lens=(batch_tokens!=self.alphabet.padding_idx).sum(axis=1)
        representations=self.model_esm(batch_tokens,repr_layers=[self.model_esm.num_layers])["representations"][self.model_esm.num_layers]
        reshaped_representations=self.norm_representation_projection(self.linear_representation_projection(representations))
        representation=th.stack([reshaped_representations[i,1:tokens_len-1].mean(axis=0) for i,tokens_len in enumerate(batch_lens)])
        return representation



class Corpus(nn.Module):
    def __init__(self,corpus):
        super().__init__()
        
        self.eps=1e-8
        self.register_buffer("flag",th.tensor(True))

        self.token_prefix="[PRE]"
        self.token_null="[NULL]"
        self.token_cls="[CLS]"
        self.token_sep="[SEP]"
        self.token_padding="[PAD]"
        self.token_mask="[MASK]"
        self.token_unknown="[UNK]"
        self.corpus_special_token=[self.token_prefix,self.token_null,self.token_cls,self.token_sep,self.token_padding,self.token_mask,self.token_unknown]

        self.corpus=corpus
        self.whole_corpus=self.corpus+self.corpus_special_token
        
        dict_token2index={}
        dict_index2token={}
        for index,token in enumerate(self.whole_corpus):
            if token in dict_token2index:
                raise Exception()
            dict_token2index[token]=index
            dict_index2token[index]=token
        self.dict_token2index=dict_token2index
        self.dict_index2token=dict_index2token

    def setEmbedding(self,d_embedding,std=0.02):
        whole_corpus_size=len(self.whole_corpus)
        self.register_parameter("embedding",nn.Parameter(th.randn(whole_corpus_size,d_embedding,dtype=th.float32)))
        init.normal_(self.embedding,0,std)

    def genEmbedding(self,token):
        device=self.flag.device
        max_n_token=max([len(token_) for token_ in token])
        embedding=[]
        for token_ in token:
            n_token=len(token_)
            n_padding=max_n_token-n_token
            token_=[token__ if token__ in self.dict_token2index else self.token_unknown for token__ in token_]+[self.token_padding]*n_padding
            index_=th.tensor([self.dict_token2index[token__] for token__ in token_],dtype=th.int64,device=device)
            embedding_=self.embedding[index_]
            embedding.append(embedding_)
        embedding=th.stack(embedding)
        return embedding
        

class GenomeMiningModel(nn.Module):
    def __init__(
        self,l_pfam_id,l_genomic_window_size,d_pfam,d_raw_predefined_appendix,d_processed_predefined_appendix,d_raw_extended_appendix,d_processed_extended_appendix,d_esm,
        d_hidden_LSTM,n_head,d_FFN,n_TFE_module_pfam_level,n_TFE_layer_pfam2pep_level,n_TFE_layer_pep_level,n_layer_LSTM,n_layer_MLP,
        activation_func,dropout_rate
    ):
        super().__init__()

        self.eps=1e-8
        self.register_buffer("flag",th.tensor(True))

        self.d_pfam=d_pfam
        self.d_raw_predefined_appendix=d_raw_predefined_appendix
        self.d_processed_predefined_appendix=d_processed_predefined_appendix
        self.d_raw_extended_appendix=d_raw_extended_appendix
        self.d_processed_extended_appendix=d_processed_extended_appendix
        self.d_model_pfam=d_pfam+d_processed_predefined_appendix+d_processed_extended_appendix
        self.d_model_esm=d_esm
        self.d_model=self.d_model_pfam+self.d_model_esm

        self.register_buffer("padding_raw_predefined_appendix",th.tensor([0]*self.d_raw_predefined_appendix,dtype=th.float32))
        self.register_parameter("prefix_processed_predefined_appendix",nn.Parameter(th.randn(self.d_processed_predefined_appendix,dtype=th.float32)))
        init.normal_(self.prefix_processed_predefined_appendix,0,0.02)
        self.register_parameter("null_processed_predefined_appendix",nn.Parameter(th.randn(self.d_processed_predefined_appendix,dtype=th.float32)))
        init.normal_(self.null_processed_predefined_appendix,0,0.02)

        self.register_buffer("padding_raw_extended_appendix",th.tensor([0]*self.d_raw_extended_appendix,dtype=th.float32))
        self.register_parameter("prefix_processed_extended_appendix",nn.Parameter(th.randn(self.d_processed_extended_appendix,dtype=th.float32)))
        init.normal_(self.prefix_processed_extended_appendix,0,0.02)
        self.register_parameter("null_processed_extended_appendix",nn.Parameter(th.randn(self.d_processed_extended_appendix,dtype=th.float32)))
        init.normal_(self.null_processed_extended_appendix,0,0.02)

        self.register_buffer("padding_pfam2pep_level",th.tensor([0]*self.d_model_pfam,dtype=th.float32))
        self.register_buffer("padding_esm",th.tensor([0]*self.d_model_esm,dtype=th.float32))

        self.corpus_pfam_id=Corpus(l_pfam_id)
        self.corpus_pfam_id.setEmbedding(d_embedding=self.d_pfam)

        self.linear_predefined_appendix_processing=nn.Linear(in_features=self.d_raw_predefined_appendix,out_features=self.d_processed_predefined_appendix)
        self.linear_extended_appendix_processing=nn.Linear(in_features=self.d_raw_extended_appendix,out_features=self.d_processed_extended_appendix)
        init.constant_(self.linear_extended_appendix_processing.weight,0)
        init.constant_(self.linear_extended_appendix_processing.bias,0)

        self.l_genomic_window_size=l_genomic_window_size
        self.n_genomic_window=len(self.l_genomic_window_size)
        
        self.n_head=n_head
        
        self.n_transformer_encoder_module_pfam_level=n_TFE_module_pfam_level
        self.l_transformer_encoder_layer_intra_pep_pfam_level=nn.ModuleList()
        self.l_transformer_encoder_layer_multi_pep_pfam_level=nn.ModuleList()
        for no_transformer_encoder_module_pfam_level in range(self.n_transformer_encoder_module_pfam_level):
            transformer_encoder_layer_intra_pep_pfam_level=TransformerEncoderLayer(d_model=self.d_model_pfam,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_intra_pep_pfam_level.append(transformer_encoder_layer_intra_pep_pfam_level)
            transformer_encoder_layer_multi_pep_pfam_level=TransformerEncoderLayer(d_model=self.d_model_pfam,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_multi_pep_pfam_level.append(transformer_encoder_layer_multi_pep_pfam_level)

        self.n_transformer_encoder_layer_pfam2pep_level=n_TFE_layer_pfam2pep_level
        self.l_transformer_encoder_layer_pfam2pep_level=nn.ModuleList()
        for no_transformer_encoder_layer_pfam2pep_level in range(self.n_transformer_encoder_layer_pfam2pep_level):
            transformer_encoder_layer_pfam2pep_level=TransformerEncoderLayer(d_model=self.d_model_pfam,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_pfam2pep_level.append(transformer_encoder_layer_pfam2pep_level)

        self.n_transformer_encoder_layer_pep_level=n_TFE_layer_pep_level
        self.l_transformer_encoder_layer_pep_level=nn.ModuleList()
        for no_transformer_encoder_layer_pep_level in range(self.n_transformer_encoder_layer_pep_level):
            transformer_encoder_layer_pep_level=TransformerEncoderLayer(d_model=self.d_model,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_pep_level.append(transformer_encoder_layer_pep_level)

        self.l_lstm=nn.ModuleList()
        self.l_lstm_norm=nn.ModuleList()
        for no_genomic_window in range(self.n_genomic_window):
            self.l_lstm.append(nn.LSTM(input_size=self.d_model,hidden_size=d_hidden_LSTM,num_layers=n_layer_LSTM,bidirectional=False,dropout=0,batch_first=True))
            self.l_lstm_norm.append(nn.LayerNorm(normalized_shape=d_hidden_LSTM))
            
        self.mlp_output=MLP(in_size=d_hidden_LSTM*self.n_genomic_window,out_size=1,n_layer=n_layer_MLP,activation_func=activation_func,dropout_rate=dropout_rate,input_dropout_off=True,output_activation_off=True)
        
        self.activation_func=activation_func

    def forward(self,l_dict_data,need_weight=False):
        device=self.flag.device

        n_sample=len(l_dict_data)
        l_n_pfam,l_n_token,l_n_pep=[],[],[]
        l_n_padding_token,l_n_padding_pep=[],[]
        token_pfam_id=[]
        embedding_raw_predefined_appendix=[]
        embedding_raw_extended_appendix=[]
        l_tensor_prefix_index=[]
        l_tensor_pfam_index=[]
        l_tensor_non_null_pfam_index=[]
        l_tensor_null_pfam_index=[]
        l_offset_tensor_prefix_index=[]
        l_offset_tensor_null_pfam_index=[]
        embedding_esm=[]
        mask_intra_pep_pfam_level,mask_multi_pep_pfam_level,mask_pfam2pep_level,mask_pep_level=[],[],[],[]
        for no_sample,dict_data in enumerate(l_dict_data):
                    
            n_pfam=dict_data["n_pfam"]
            n_token=dict_data["n_token"]
            n_pep=dict_data["n_pep"]
            last_pep_index=(n_pep-1)
            l_n_pfam.append(n_pfam)
            l_n_token.append(n_token)
            l_n_pep.append(n_pep)
            n_padding_token=dict_data["n_padding_token"]
            n_padding_pep=dict_data["n_padding_pep"]
            l_n_padding_token.append(n_padding_token)
            l_n_padding_pep.append(n_padding_pep)
            max_n_pfam_within_batch=dict_data["max_n_pfam_within_batch"]
            max_n_token_within_batch=dict_data["max_n_token_within_batch"]
            max_n_pep_within_batch=dict_data["max_n_pep_within_batch"]
        
            token_pfam_id_=dict_data["token_pfam_id"]
            embedding_raw_predefined_appendix_=th.concat([th.tensor(dict_data["embedding_raw_predefined_appendix"],dtype=th.float32,device=device),self.padding_raw_predefined_appendix.repeat([n_padding_token,1])],axis=0)
            embedding_raw_extended_appendix_=th.concat([th.tensor(dict_data["embedding_raw_extended_appendix"],dtype=th.float32,device=device),self.padding_raw_extended_appendix.repeat([n_padding_token,1])],axis=0)

            prefix_index=dict_data["prefix_index"]
            tensor_prefix_index=th.tensor(prefix_index,dtype=th.int64,device=device)
            pfam_index=dict_data["pfam_index"]
            tensor_pfam_index=th.tensor(pfam_index,dtype=th.int64,device=device)
            non_null_pfam_index=dict_data["non_null_pfam_index"]
            tensor_non_null_pfam_index=th.tensor(non_null_pfam_index,dtype=th.int64,device=device)
            null_pfam_index=dict_data["null_pfam_index"]
            tensor_null_pfam_index=th.tensor(null_pfam_index,dtype=th.int64,device=device)
            
            offset=no_sample*max_n_token_within_batch
            offset_tensor_prefix_index=tensor_prefix_index+offset
            offset_tensor_null_pfam_index=tensor_null_pfam_index+offset
            
            dict_prefix_index2pfam_index_block=dict_data["dict_prefix_index2pfam_index_block"]
            dict_pfam_index2prefix_index=dict_data["dict_pfam_index2prefix_index"]
            dict_pfam_index2pfam_index_block=dict_data["dict_pfam_index2pfam_index_block"]
        
            token_pfam_id.append(token_pfam_id_)
            embedding_raw_predefined_appendix.append(embedding_raw_predefined_appendix_)
            embedding_raw_extended_appendix.append(embedding_raw_extended_appendix_)
            l_tensor_prefix_index.append(tensor_prefix_index)
            l_tensor_pfam_index.append(tensor_pfam_index)
            l_tensor_non_null_pfam_index.append(tensor_non_null_pfam_index)
            l_tensor_null_pfam_index.append(tensor_null_pfam_index)
            l_offset_tensor_prefix_index.append(offset_tensor_prefix_index)
            l_offset_tensor_null_pfam_index.append(offset_tensor_null_pfam_index)
        
            embedding_esm_=th.concat([th.tensor(dict_data["embedding_esm"],dtype=th.float32,device=device),self.padding_esm.repeat([n_padding_pep,1])],axis=0)
            embedding_esm.append(embedding_esm_)
        
            original_mask_intra_pep_pfam_level_=np.ones([n_token,n_token]).astype(int)
            original_mask_intra_pep_pfam_level_[np.diag_indices_from(original_mask_intra_pep_pfam_level_)]=0
            for pfam_index_ in pfam_index:
                pfam_index_block=dict_pfam_index2pfam_index_block[pfam_index_]
                original_mask_intra_pep_pfam_level_[pfam_index_,pfam_index_block]=0
            for prefix_index_ in prefix_index:
                pfam_index_block=dict_prefix_index2pfam_index_block[prefix_index_]
                original_mask_intra_pep_pfam_level_[prefix_index_,pfam_index_block]=0
            mask_intra_pep_pfam_level_=np.ones([max_n_token_within_batch,max_n_token_within_batch]).astype(int)
            mask_intra_pep_pfam_level_[np.diag_indices_from(mask_intra_pep_pfam_level_)]=0
            mask_intra_pep_pfam_level_[:n_token,:n_token]=original_mask_intra_pep_pfam_level_
            mask_intra_pep_pfam_level_=th.tensor(mask_intra_pep_pfam_level_,dtype=th.bool,device=device)

            mask_multi_pep_pfam_level_=[]
            for genomic_window_size in self.l_genomic_window_size:
                original_mask_multi_pep_pfam_level__=np.ones([n_token,n_token]).astype(int)
                original_mask_multi_pep_pfam_level__[np.diag_indices_from(original_mask_multi_pep_pfam_level__)]=0
                for current_pep_index in range(n_pep):
                    prefix_index_=prefix_index[current_pep_index]
                    pfam_index_block=dict_prefix_index2pfam_index_block[prefix_index_]
                    neighbor_pep_index_start=max((current_pep_index-genomic_window_size),0)
                    neighbor_pep_index_end=min((current_pep_index+genomic_window_size),last_pep_index)
                    neighbor_pfam_index=[]
                    for neighbor_pep_index in range(neighbor_pep_index_start,(neighbor_pep_index_end+1)):
                        neighbor_prefix_index=prefix_index[neighbor_pep_index]
                        neighbor_pfam_index_block=dict_prefix_index2pfam_index_block[neighbor_prefix_index]
                        neighbor_pfam_index+=neighbor_pfam_index_block
                    for pfam_index_ in pfam_index_block:
                        original_mask_multi_pep_pfam_level__[pfam_index_,neighbor_pfam_index]=0
                    original_mask_multi_pep_pfam_level__[prefix_index_,pfam_index_block]=0
                mask_multi_pep_pfam_level__=np.ones([max_n_token_within_batch,max_n_token_within_batch]).astype(int)
                mask_multi_pep_pfam_level__[np.diag_indices_from(mask_multi_pep_pfam_level__)]=0
                mask_multi_pep_pfam_level__[:n_token,:n_token]=original_mask_multi_pep_pfam_level__
                mask_multi_pep_pfam_level__=th.tensor(mask_multi_pep_pfam_level__,dtype=th.bool,device=device)
                mask_multi_pep_pfam_level_.append(mask_multi_pep_pfam_level__)
            mask_multi_pep_pfam_level_=th.stack(mask_multi_pep_pfam_level_)

            original_mask_pfam2pep_level_=np.ones([n_token,n_token]).astype(int)
            original_mask_pfam2pep_level_[np.diag_indices_from(original_mask_pfam2pep_level_)]=0
            for prefix_index_ in prefix_index:
                pfam_index_block=dict_prefix_index2pfam_index_block[prefix_index_]
                original_mask_pfam2pep_level_[prefix_index_,pfam_index_block]=0
            mask_pfam2pep_level_=np.ones([max_n_token_within_batch,max_n_token_within_batch]).astype(int)
            mask_pfam2pep_level_[np.diag_indices_from(mask_pfam2pep_level_)]=0
            mask_pfam2pep_level_[:n_token,:n_token]=original_mask_pfam2pep_level_
            mask_pfam2pep_level_=th.tensor(mask_pfam2pep_level_,dtype=th.bool,device=device)

            mask_pep_level_=[]
            for genomic_window_size in self.l_genomic_window_size:
                original_mask_pep_level__=np.ones([n_pep,n_pep]).astype(int)
                original_mask_pep_level__[np.diag_indices_from(original_mask_pep_level__)]=0
                for current_pep_index in range(n_pep):
                    neighbor_pep_index_start=max((current_pep_index-genomic_window_size),0)
                    neighbor_pep_index_end=min((current_pep_index+genomic_window_size),last_pep_index)
                    original_mask_pep_level__[current_pep_index,neighbor_pep_index_start:(neighbor_pep_index_end+1)]=0
                mask_pep_level__=np.ones([max_n_pep_within_batch,max_n_pep_within_batch]).astype(int)
                mask_pep_level__[np.diag_indices_from(mask_pep_level__)]=0
                mask_pep_level__[:n_pep,:n_pep]=original_mask_pep_level__
                mask_pep_level__=th.tensor(mask_pep_level__,dtype=th.bool,device=device)
                mask_pep_level_.append(mask_pep_level__)
            mask_pep_level_=th.stack(mask_pep_level_)

            mask_intra_pep_pfam_level.append(th.stack([mask_intra_pep_pfam_level_]*self.n_head))
            mask_multi_pep_pfam_level.append(th.stack([mask_multi_pep_pfam_level_]*self.n_head))
            mask_pfam2pep_level.append(th.stack([mask_pfam2pep_level_]*self.n_head))
            mask_pep_level.append(th.stack([mask_pep_level_]*self.n_head))
        
        embedding_pfam_id=self.corpus_pfam_id.genEmbedding(token_pfam_id)
        embedding_raw_predefined_appendix=th.concat(embedding_raw_predefined_appendix,axis=0)
        embedding_processed_predefined_appendix=self.linear_predefined_appendix_processing(embedding_raw_predefined_appendix)
        embedding_processed_predefined_appendix[th.concat(l_offset_tensor_prefix_index)]=self.prefix_processed_predefined_appendix
        embedding_processed_predefined_appendix[th.concat(l_offset_tensor_null_pfam_index)]=self.null_processed_predefined_appendix
        embedding_processed_predefined_appendix=embedding_processed_predefined_appendix.reshape(n_sample,max_n_token_within_batch,self.d_processed_predefined_appendix)
        embedding_raw_extended_appendix=th.concat(embedding_raw_extended_appendix,axis=0)
        embedding_processed_extended_appendix=self.linear_extended_appendix_processing(embedding_raw_extended_appendix)
        embedding_processed_extended_appendix[th.concat(l_offset_tensor_prefix_index)]=self.prefix_processed_extended_appendix
        embedding_processed_extended_appendix[th.concat(l_offset_tensor_null_pfam_index)]=self.null_processed_extended_appendix
        embedding_processed_extended_appendix=embedding_processed_extended_appendix.reshape(n_sample,max_n_token_within_batch,self.d_processed_extended_appendix)
        embedding_esm=th.stack(embedding_esm)
        
        mask_intra_pep_pfam_level=th.stack(mask_intra_pep_pfam_level)
        mask_multi_pep_pfam_level=th.stack(mask_multi_pep_pfam_level).permute(2,0,1,3,4)
        mask_pfam2pep_level=th.stack(mask_pfam2pep_level)
        mask_pep_level=th.stack(mask_pep_level).permute(2,0,1,3,4)

        x=th.concat([embedding_pfam_id,embedding_processed_predefined_appendix,embedding_processed_extended_appendix],axis=2)
        y=[]
        ll_attn_weight_intra_pep_pfam_level,ll_attn_weight_multi_pep_pfam_level=[],[]
        ll_attn_weight_pfam2pep_level=[]
        ll_attn_weight_pep_level=[]
        for no_genomic_window,genomic_window_size in enumerate(self.l_genomic_window_size):
            x_=x
            l_attn_weight_intra_pep_pfam_level,l_attn_weight_multi_pep_pfam_level=[],[]
            for no_transformer_encoder_module_pfam_level in range(self.n_transformer_encoder_module_pfam_level):
                transformer_encoder_layer_intra_pep_pfam_level=self.l_transformer_encoder_layer_intra_pep_pfam_level[no_transformer_encoder_module_pfam_level]
                x_,attn_weight_intra_pep_pfam_level=transformer_encoder_layer_intra_pep_pfam_level(x_,attn_mask=mask_intra_pep_pfam_level,need_weight=need_weight)
                l_attn_weight_intra_pep_pfam_level.append(attn_weight_intra_pep_pfam_level)
                transformer_encoder_layer_multi_pep_pfam_level=self.l_transformer_encoder_layer_multi_pep_pfam_level[no_transformer_encoder_module_pfam_level]
                x_,attn_weight_multi_pep_pfam_level=transformer_encoder_layer_multi_pep_pfam_level(x_,attn_mask=mask_multi_pep_pfam_level[no_genomic_window,:,:,:,:],need_weight=need_weight)
                l_attn_weight_multi_pep_pfam_level.append(attn_weight_multi_pep_pfam_level)

            l_attn_weight_pfam2pep_level=[]
            for no_transformer_encoder_layer_pfam2pep_level in range(self.n_transformer_encoder_layer_pfam2pep_level):
                transformer_encoder_layer_pfam2pep_level=self.l_transformer_encoder_layer_pfam2pep_level[no_transformer_encoder_layer_pfam2pep_level]
                x_,attn_weight_pfam2pep_level=transformer_encoder_layer_pfam2pep_level(x_,attn_mask=mask_pfam2pep_level,need_weight=need_weight)
                l_attn_weight_pfam2pep_level.append(attn_weight_pfam2pep_level)
            
            x_=th.stack([
                th.concat(
                    [x__[tensor_prefix_index],self.padding_pfam2pep_level.repeat([n_padding_pep,1])],axis=0
                ) for x__,tensor_prefix_index,n_padding_pep in zip(x_,l_tensor_prefix_index,l_n_padding_pep)]
            )
            x_=th.concat([x_,embedding_esm],axis=2)

            l_attn_weight_pep_level=[]
            for no_transformer_encoder_layer_pep_level in range(self.n_transformer_encoder_layer_pep_level):
                transformer_encoder_layer_pep_level=self.l_transformer_encoder_layer_pep_level[no_transformer_encoder_layer_pep_level]
                x_,attn_weight_pep_level=transformer_encoder_layer_pep_level(x_,attn_mask=mask_pep_level[no_genomic_window,:,:,:,:],need_weight=need_weight)
                l_attn_weight_pep_level.append(attn_weight_pep_level)

            y.append(x_)
            ll_attn_weight_intra_pep_pfam_level.append(l_attn_weight_intra_pep_pfam_level)
            ll_attn_weight_multi_pep_pfam_level.append(l_attn_weight_multi_pep_pfam_level)
            ll_attn_weight_pfam2pep_level.append(l_attn_weight_pfam2pep_level)
            ll_attn_weight_pep_level.append(l_attn_weight_pep_level)
            
        y=th.stack(y).permute(1,0,2,3)
        y=[y_[:,:n_pep,:] for y_,n_pep in zip(y,l_n_pep)]
        y=[th.concat([lstm_norm(lstm(y_[no_genomic_window,:,:])[0]+lstm(y_[no_genomic_window,:,:].flip(0))[0].flip(0)) for no_genomic_window,(lstm,lstm_norm) in enumerate(zip(self.l_lstm,self.l_lstm_norm))],axis=1) for y_ in y]
        pred_target=[th.sigmoid(self.mlp_output(y_).squeeze(1)) for y_ in y]
        return pred_target,(ll_attn_weight_intra_pep_pfam_level,ll_attn_weight_multi_pep_pfam_level,ll_attn_weight_pfam2pep_level,ll_attn_weight_pep_level)


class CoreEnzymeIdentificationModel(nn.Module):
    def __init__(
        self,l_pfam_id,d_pfam,d_raw_predefined_appendix,d_processed_predefined_appendix,d_raw_extended_appendix,d_processed_extended_appendix,d_esm,
        n_head,d_FFN,n_TFE_module_pfam_level,n_TFE_layer_pfam2pep_level,n_TFE_layer_pep_level,n_layer_MLP,
        activation_func,dropout_rate
    ):
        super().__init__()

        self.eps=1e-8
        self.register_buffer("flag",th.tensor(True))

        self.d_pfam=d_pfam
        self.d_raw_predefined_appendix=d_raw_predefined_appendix
        self.d_processed_predefined_appendix=d_processed_predefined_appendix
        self.d_raw_extended_appendix=d_raw_extended_appendix
        self.d_processed_extended_appendix=d_processed_extended_appendix
        self.d_model_pfam=d_pfam+d_processed_predefined_appendix+d_processed_extended_appendix
        self.d_model_esm=d_esm
        self.d_model=self.d_model_pfam+self.d_model_esm

        self.register_buffer("padding_raw_predefined_appendix",th.tensor([0]*self.d_raw_predefined_appendix,dtype=th.float32))
        self.register_parameter("prefix_processed_predefined_appendix",nn.Parameter(th.randn(self.d_processed_predefined_appendix,dtype=th.float32)))
        init.normal_(self.prefix_processed_predefined_appendix,0,0.02)
        self.register_parameter("null_processed_predefined_appendix",nn.Parameter(th.randn(self.d_processed_predefined_appendix,dtype=th.float32)))
        init.normal_(self.null_processed_predefined_appendix,0,0.02)

        self.register_buffer("padding_raw_extended_appendix",th.tensor([0]*self.d_raw_extended_appendix,dtype=th.float32))
        self.register_parameter("prefix_processed_extended_appendix",nn.Parameter(th.randn(self.d_processed_extended_appendix,dtype=th.float32)))
        init.normal_(self.prefix_processed_extended_appendix,0,0.02)
        self.register_parameter("null_processed_extended_appendix",nn.Parameter(th.randn(self.d_processed_extended_appendix,dtype=th.float32)))
        init.normal_(self.null_processed_extended_appendix,0,0.02)

        self.register_buffer("padding_pfam2pep_level",th.tensor([0]*self.d_model_pfam,dtype=th.float32))
        self.register_buffer("padding_esm",th.tensor([0]*self.d_model_esm,dtype=th.float32))

        self.corpus_pfam_id=Corpus(l_pfam_id)
        self.corpus_pfam_id.setEmbedding(d_embedding=self.d_pfam)

        self.linear_predefined_appendix_processing=nn.Linear(in_features=self.d_raw_predefined_appendix,out_features=self.d_processed_predefined_appendix)
        self.linear_extended_appendix_processing=nn.Linear(in_features=self.d_raw_extended_appendix,out_features=self.d_processed_extended_appendix)
        init.constant_(self.linear_extended_appendix_processing.weight,0)
        init.constant_(self.linear_extended_appendix_processing.bias,0)
        
        self.n_head=n_head
        
        self.n_transformer_encoder_module_pfam_level=n_TFE_module_pfam_level
        self.l_transformer_encoder_layer_intra_pep_pfam_level=nn.ModuleList()
        self.l_transformer_encoder_layer_multi_pep_pfam_level=nn.ModuleList()
        for no_transformer_encoder_module_pfam_level in range(self.n_transformer_encoder_module_pfam_level):
            transformer_encoder_layer_intra_pep_pfam_level=TransformerEncoderLayer(d_model=self.d_model_pfam,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_intra_pep_pfam_level.append(transformer_encoder_layer_intra_pep_pfam_level)
            transformer_encoder_layer_multi_pep_pfam_level=TransformerEncoderLayer(d_model=self.d_model_pfam,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_multi_pep_pfam_level.append(transformer_encoder_layer_multi_pep_pfam_level)

        self.n_transformer_encoder_layer_pfam2pep_level=n_TFE_layer_pfam2pep_level
        self.l_transformer_encoder_layer_pfam2pep_level=nn.ModuleList()
        for no_transformer_encoder_layer_pfam2pep_level in range(self.n_transformer_encoder_layer_pfam2pep_level):
            transformer_encoder_layer_pfam2pep_level=TransformerEncoderLayer(d_model=self.d_model_pfam,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_pfam2pep_level.append(transformer_encoder_layer_pfam2pep_level)

        self.n_transformer_encoder_layer_pep_level=n_TFE_layer_pep_level
        self.l_transformer_encoder_layer_pep_level=nn.ModuleList()
        for no_transformer_encoder_layer_pep_level in range(self.n_transformer_encoder_layer_pep_level):
            transformer_encoder_layer_pep_level=TransformerEncoderLayer(d_model=self.d_model,nhead=n_head,dim_feedforward=d_FFN,activation=activation_func,dropout=dropout_rate)
            self.l_transformer_encoder_layer_pep_level.append(transformer_encoder_layer_pep_level)
            
        self.mlp_output=MLP(in_size=self.d_model,out_size=1,n_layer=n_layer_MLP,activation_func=activation_func,dropout_rate=dropout_rate,input_dropout_off=True,output_activation_off=True)
        
        self.activation_func=activation_func

    def forward(self,l_dict_data,need_weight=False):
        device=self.flag.device

        n_sample=len(l_dict_data)
        l_n_pfam,l_n_token,l_n_pep=[],[],[]
        l_n_padding_token,l_n_padding_pep=[],[]
        token_pfam_id=[]
        embedding_raw_predefined_appendix=[]
        embedding_raw_extended_appendix=[]
        l_tensor_prefix_index=[]
        l_tensor_pfam_index=[]
        l_tensor_non_null_pfam_index=[]
        l_tensor_null_pfam_index=[]
        l_offset_tensor_prefix_index=[]
        l_offset_tensor_null_pfam_index=[]
        embedding_esm=[]
        mask_intra_pep_pfam_level,mask_multi_pep_pfam_level,mask_pfam2pep_level,mask_pep_level=[],[],[],[]
        for no_sample,dict_data in enumerate(l_dict_data):
                    
            n_pfam=dict_data["n_pfam"]
            n_token=dict_data["n_token"]
            n_pep=dict_data["n_pep"]
            last_pep_index=(n_pep-1)
            l_n_pfam.append(n_pfam)
            l_n_token.append(n_token)
            l_n_pep.append(n_pep)
            n_padding_token=dict_data["n_padding_token"]
            n_padding_pep=dict_data["n_padding_pep"]
            l_n_padding_token.append(n_padding_token)
            l_n_padding_pep.append(n_padding_pep)
            max_n_pfam_within_batch=dict_data["max_n_pfam_within_batch"]
            max_n_token_within_batch=dict_data["max_n_token_within_batch"]
            max_n_pep_within_batch=dict_data["max_n_pep_within_batch"]
        
            token_pfam_id_=dict_data["token_pfam_id"]
            embedding_raw_predefined_appendix_=th.concat([th.tensor(dict_data["embedding_raw_predefined_appendix"],dtype=th.float32,device=device),self.padding_raw_predefined_appendix.repeat([n_padding_token,1])],axis=0)
            embedding_raw_extended_appendix_=th.concat([th.tensor(dict_data["embedding_raw_extended_appendix"],dtype=th.float32,device=device),self.padding_raw_extended_appendix.repeat([n_padding_token,1])],axis=0)

            prefix_index=dict_data["prefix_index"]
            tensor_prefix_index=th.tensor(prefix_index,dtype=th.int64,device=device)
            pfam_index=dict_data["pfam_index"]
            tensor_pfam_index=th.tensor(pfam_index,dtype=th.int64,device=device)
            non_null_pfam_index=dict_data["non_null_pfam_index"]
            tensor_non_null_pfam_index=th.tensor(non_null_pfam_index,dtype=th.int64,device=device)
            null_pfam_index=dict_data["null_pfam_index"]
            tensor_null_pfam_index=th.tensor(null_pfam_index,dtype=th.int64,device=device)
            
            offset=no_sample*max_n_token_within_batch
            offset_tensor_prefix_index=tensor_prefix_index+offset
            offset_tensor_null_pfam_index=tensor_null_pfam_index+offset
            
            dict_prefix_index2pfam_index_block=dict_data["dict_prefix_index2pfam_index_block"]
            dict_pfam_index2prefix_index=dict_data["dict_pfam_index2prefix_index"]
            dict_pfam_index2pfam_index_block=dict_data["dict_pfam_index2pfam_index_block"]
        
            token_pfam_id.append(token_pfam_id_)
            embedding_raw_predefined_appendix.append(embedding_raw_predefined_appendix_)
            embedding_raw_extended_appendix.append(embedding_raw_extended_appendix_)
            l_tensor_prefix_index.append(tensor_prefix_index)
            l_tensor_pfam_index.append(tensor_pfam_index)
            l_tensor_non_null_pfam_index.append(tensor_non_null_pfam_index)
            l_tensor_null_pfam_index.append(tensor_null_pfam_index)
            l_offset_tensor_prefix_index.append(offset_tensor_prefix_index)
            l_offset_tensor_null_pfam_index.append(offset_tensor_null_pfam_index)
        
            embedding_esm_=th.concat([th.tensor(dict_data["embedding_esm"],dtype=th.float32,device=device),self.padding_esm.repeat([n_padding_pep,1])],axis=0)
            embedding_esm.append(embedding_esm_)
        
            original_mask_intra_pep_pfam_level_=np.ones([n_token,n_token]).astype(int)
            original_mask_intra_pep_pfam_level_[np.diag_indices_from(original_mask_intra_pep_pfam_level_)]=0
            for pfam_index_ in pfam_index:
                pfam_index_block=dict_pfam_index2pfam_index_block[pfam_index_]
                original_mask_intra_pep_pfam_level_[pfam_index_,pfam_index_block]=0
            for prefix_index_ in prefix_index:
                pfam_index_block=dict_prefix_index2pfam_index_block[prefix_index_]
                original_mask_intra_pep_pfam_level_[prefix_index_,pfam_index_block]=0
            mask_intra_pep_pfam_level_=np.ones([max_n_token_within_batch,max_n_token_within_batch]).astype(int)
            mask_intra_pep_pfam_level_[np.diag_indices_from(mask_intra_pep_pfam_level_)]=0
            mask_intra_pep_pfam_level_[:n_token,:n_token]=original_mask_intra_pep_pfam_level_
            mask_intra_pep_pfam_level_=th.tensor(mask_intra_pep_pfam_level_,dtype=th.bool,device=device)

            original_mask_multi_pep_pfam_level_=np.ones([n_token,n_token]).astype(int)
            original_mask_multi_pep_pfam_level_[np.diag_indices_from(original_mask_multi_pep_pfam_level_)]=0
            for pfam_index_ in pfam_index:
                original_mask_multi_pep_pfam_level_[pfam_index_,pfam_index]=0
            for prefix_index_ in prefix_index:
                pfam_index_block=dict_prefix_index2pfam_index_block[prefix_index_]
                original_mask_multi_pep_pfam_level_[prefix_index_,pfam_index_block]=0
            mask_multi_pep_pfam_level_=np.ones([max_n_token_within_batch,max_n_token_within_batch]).astype(int)
            mask_multi_pep_pfam_level_[np.diag_indices_from(mask_multi_pep_pfam_level_)]=0
            mask_multi_pep_pfam_level_[:n_token,:n_token]=original_mask_multi_pep_pfam_level_
            mask_multi_pep_pfam_level_=th.tensor(mask_multi_pep_pfam_level_,dtype=th.bool,device=device)

            original_mask_pfam2pep_level_=np.ones([n_token,n_token]).astype(int)
            original_mask_pfam2pep_level_[np.diag_indices_from(original_mask_pfam2pep_level_)]=0
            for prefix_index_ in prefix_index:
                pfam_index_block=dict_prefix_index2pfam_index_block[prefix_index_]
                original_mask_pfam2pep_level_[prefix_index_,pfam_index_block]=0
            mask_pfam2pep_level_=np.ones([max_n_token_within_batch,max_n_token_within_batch]).astype(int)
            mask_pfam2pep_level_[np.diag_indices_from(mask_pfam2pep_level_)]=0
            mask_pfam2pep_level_[:n_token,:n_token]=original_mask_pfam2pep_level_
            mask_pfam2pep_level_=th.tensor(mask_pfam2pep_level_,dtype=th.bool,device=device)

            original_mask_pep_level_=np.zeros([n_pep,n_pep]).astype(int)
            mask_pep_level_=np.ones([max_n_pep_within_batch,max_n_pep_within_batch]).astype(int)
            mask_pep_level_[np.diag_indices_from(mask_pep_level_)]=0
            mask_pep_level_[:n_pep,:n_pep]=original_mask_pep_level_
            mask_pep_level_=th.tensor(mask_pep_level_,dtype=th.bool,device=device)

            mask_intra_pep_pfam_level.append(th.stack([mask_intra_pep_pfam_level_]*self.n_head))
            mask_multi_pep_pfam_level.append(th.stack([mask_multi_pep_pfam_level_]*self.n_head))
            mask_pfam2pep_level.append(th.stack([mask_pfam2pep_level_]*self.n_head))
            mask_pep_level.append(th.stack([mask_pep_level_]*self.n_head))
        
        embedding_pfam_id=self.corpus_pfam_id.genEmbedding(token_pfam_id)
        embedding_raw_predefined_appendix=th.concat(embedding_raw_predefined_appendix,axis=0)
        embedding_processed_predefined_appendix=self.linear_predefined_appendix_processing(embedding_raw_predefined_appendix)
        embedding_processed_predefined_appendix[th.concat(l_offset_tensor_prefix_index)]=self.prefix_processed_predefined_appendix
        embedding_processed_predefined_appendix[th.concat(l_offset_tensor_null_pfam_index)]=self.null_processed_predefined_appendix
        embedding_processed_predefined_appendix=embedding_processed_predefined_appendix.reshape(n_sample,max_n_token_within_batch,self.d_processed_predefined_appendix)
        embedding_raw_extended_appendix=th.concat(embedding_raw_extended_appendix,axis=0)
        embedding_processed_extended_appendix=self.linear_extended_appendix_processing(embedding_raw_extended_appendix)
        embedding_processed_extended_appendix[th.concat(l_offset_tensor_prefix_index)]=self.prefix_processed_extended_appendix
        embedding_processed_extended_appendix[th.concat(l_offset_tensor_null_pfam_index)]=self.null_processed_extended_appendix
        embedding_processed_extended_appendix=embedding_processed_extended_appendix.reshape(n_sample,max_n_token_within_batch,self.d_processed_extended_appendix)
        embedding_esm=th.stack(embedding_esm)
        
        mask_intra_pep_pfam_level=th.stack(mask_intra_pep_pfam_level)
        mask_multi_pep_pfam_level=th.stack(mask_multi_pep_pfam_level)
        mask_pfam2pep_level=th.stack(mask_pfam2pep_level)
        mask_pep_level=th.stack(mask_pep_level)

        x=th.concat([embedding_pfam_id,embedding_processed_predefined_appendix,embedding_processed_extended_appendix],axis=2)
        ll_attn_weight_intra_pep_pfam_level,ll_attn_weight_multi_pep_pfam_level=[],[]
        ll_attn_weight_pfam2pep_level=[]
        ll_attn_weight_pep_level=[]

        l_attn_weight_intra_pep_pfam_level,l_attn_weight_multi_pep_pfam_level=[],[]
        for no_transformer_encoder_module_pfam_level in range(self.n_transformer_encoder_module_pfam_level):
            transformer_encoder_layer_intra_pep_pfam_level=self.l_transformer_encoder_layer_intra_pep_pfam_level[no_transformer_encoder_module_pfam_level]
            x,attn_weight_intra_pep_pfam_level=transformer_encoder_layer_intra_pep_pfam_level(x,attn_mask=mask_intra_pep_pfam_level,need_weight=need_weight)
            l_attn_weight_intra_pep_pfam_level.append(attn_weight_intra_pep_pfam_level)
            transformer_encoder_layer_multi_pep_pfam_level=self.l_transformer_encoder_layer_multi_pep_pfam_level[no_transformer_encoder_module_pfam_level]
            x,attn_weight_multi_pep_pfam_level=transformer_encoder_layer_multi_pep_pfam_level(x,attn_mask=mask_multi_pep_pfam_level,need_weight=need_weight)
            l_attn_weight_multi_pep_pfam_level.append(attn_weight_multi_pep_pfam_level)

        l_attn_weight_pfam2pep_level=[]
        for no_transformer_encoder_layer_pfam2pep_level in range(self.n_transformer_encoder_layer_pfam2pep_level):
            transformer_encoder_layer_pfam2pep_level=self.l_transformer_encoder_layer_pfam2pep_level[no_transformer_encoder_layer_pfam2pep_level]
            x,attn_weight_pfam2pep_level=transformer_encoder_layer_pfam2pep_level(x,attn_mask=mask_pfam2pep_level,need_weight=need_weight)
            l_attn_weight_pfam2pep_level.append(attn_weight_pfam2pep_level)
        
        x=th.stack([
            th.concat(
                [x_[tensor_prefix_index],self.padding_pfam2pep_level.repeat([n_padding_pep,1])],axis=0
            ) for x_,tensor_prefix_index,n_padding_pep in zip(x,l_tensor_prefix_index,l_n_padding_pep)]
        )
        x=th.concat([x,embedding_esm],axis=2)

        l_attn_weight_pep_level=[]
        for no_transformer_encoder_layer_pep_level in range(self.n_transformer_encoder_layer_pep_level):
            transformer_encoder_layer_pep_level=self.l_transformer_encoder_layer_pep_level[no_transformer_encoder_layer_pep_level]
            x,attn_weight_pep_level=transformer_encoder_layer_pep_level(x,attn_mask=mask_pep_level,need_weight=need_weight)
            l_attn_weight_pep_level.append(attn_weight_pep_level)

        ll_attn_weight_intra_pep_pfam_level.append(l_attn_weight_intra_pep_pfam_level)
        ll_attn_weight_multi_pep_pfam_level.append(l_attn_weight_multi_pep_pfam_level)
        ll_attn_weight_pfam2pep_level.append(l_attn_weight_pfam2pep_level)
        ll_attn_weight_pep_level.append(l_attn_weight_pep_level)
            
        x=[x_[:n_pep,:] for x_,n_pep in zip(x,l_n_pep)]
        pred_target=[th.sigmoid(self.mlp_output(x_).squeeze(1)) for x_ in x]
        return pred_target,(ll_attn_weight_intra_pep_pfam_level,ll_attn_weight_multi_pep_pfam_level,ll_attn_weight_pfam2pep_level,ll_attn_weight_pep_level)