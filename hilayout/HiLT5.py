import random, warnings
import re
import numpy as np
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm as BertLayerNorm

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput, BaseModelOutput
from hilayout._modules import CustomT5Config, SpatialEmbeddings, RetrievalModule
import transformers.models.t5.modeling_t5
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

# class HiLT5Config(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`HiLT5`]. It is used to
#     instantiate a HiLT5 model according to the specified arguments, defining the model architecture

#     Args:
#         max_position_embeddings
#             Trained max position embeddings in the Donut decoder,
#             if not specified, it will have same value with max_length
#         max_length:
#             Max position embeddings(=maximum sequence length) you want to train
#         name_or_path:
#             Name of a pretrained model name either registered in huggingface.co. or saved in local
#     """

#     model_type = "hilt5"

#     def __init__(
#         self,
#         save_dir: str = "save/",
#         model_name: str = "Hi-LT5",
#         freeze_encoder: bool = False,
#         page_tokens: int = 10,
#         device: str = "cuda",
#         data_parallel: bool = False,
#         batch_size: int = 2,
#         max_position_embeddings: int = None,
#         max_length: int = 1536,
#         name_or_path: Union[str, bytes, os.PathLike] = "",
#         **kwargs,
#     ):
#         super().__init__()
#         self.max_position_embeddings = (
#             max_length
#             if max_position_embeddings is None
#             else max_position_embeddings
#         )
#         self.max_length = max_length
#         self.name_or_path = name_or_path



class HiLT5(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.spatial_embeddings = SpatialEmbeddings(config)
        # self.retrieval_module = RetrievalModule(config)

        self.page_tokens = config.page_tokens
        self.max_doc_pages = config.max_doc_pages

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        extra_kwargs_to_be_removed = ['bbox', 'attention_mask', 'num_pages']
        encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any(argument.startswith(p) for p in irrelevant_prefix + extra_kwargs_to_be_removed)}

        # 2.2 replace input ids by the hierarchical layout-aware input embeddings
        page_embeddings = []
        for p_idx in range(max(model_kwargs['num_pages'])):
            semantic_emb = self.shared(inputs_tensor[:, p_idx])  # read from default T5
            spatial_emb = self.spatial_embeddings(model_kwargs['bbox'][:, p_idx])
            inputs_embeds = semantic_emb + spatial_emb

            encoder_outputs = encoder(
                input_ids=None,
                attention_mask=model_kwargs['attention_mask'][:, p_idx],
                inputs_embeds=inputs_embeds,
                **encoder_kwargs
            )

            hidden_states = encoder_outputs[0]
            page_embeddings.append(hidden_states[:, :self.page_tokens])

        document_embeddings = torch.cat(page_embeddings, dim=1)

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = None
        # model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"]: ModelOutput = ModelOutput({'last_hidden_state': document_embeddings})
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "num_pages": kwargs.get('num_pages'),
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        num_pages=None,
        answer_page_idx=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            page_embeddings = []
            page_encoder_attentions = []
            # for page_idx in range(self.max_doc_pages):
            for page_idx in range(max(num_pages)):
                semantic_emb = self.shared(input_ids[:, page_idx])  # read from default T5
                # spatial_emb = self.emb_matcher(self.spatial_embeddings(bbox[:, page_idx]))
                spatial_emb = self.spatial_embeddings(bbox[:, page_idx])
                inputs_embeds = semantic_emb + spatial_emb
                encoder_outputs = self.encoder(
                    input_ids=None,  # Input IDs must be None because input embeds is provided.
                    attention_mask=attention_mask[:, page_idx],
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                # Keep only [PAGE] token representation.
                hidden_states = encoder_outputs[0]
                page_embeddings.append(hidden_states[:, :self.page_tokens])

                if output_attentions:
                    page_encoder_attentions.append(encoder_outputs.attentions)

            document_embeddings = torch.cat(page_embeddings, dim=1)

            # attention_mask = torch.zeros([hidden_states.shape[0], self.num_doc_cls_tokens * self.doc_pages]).to(document_embeddings.device)  # Pages, hidden size. Make use of all information of the document embedding
            attention_mask = torch.zeros([hidden_states.shape[0], self.page_tokens * max(num_pages)]).to(document_embeddings.device)  # Pages, hidden size. Make use of all information of the document embedding
            for bs_idx in range(len(hidden_states)):
                attention_mask[bs_idx, :min(num_pages[bs_idx], self.max_doc_pages) * self.page_tokens] = 1

            attention_mask = attention_mask.to(document_embeddings.device)

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):  # EncoderOutputs is True when comes from _prepare_encoder_decoder_kwargs_for_generation, during .generation function.
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

            hidden_states = encoder_outputs[0]  # TODO - This should be replaced by document embeddings
            # TODO - Create the Multipage mask.

            """  ==== NEW ==== """
            document_embeddings = hidden_states

            attention_mask = torch.zeros([hidden_states.shape[0], self.page_tokens * max(num_pages)])
            for bs_idx in range(len(hidden_states)):
                attention_mask[bs_idx, : min(num_pages[bs_idx], max(num_pages)) * self.page_tokens] = 1

            attention_mask = attention_mask.to(document_embeddings.device)
            """  ==== END NEW ==== """

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            # encoder_hidden_states=hidden_states,
            encoder_hidden_states=document_embeddings,  # Previous 'hidden states' in original T5
            encoder_attention_mask=attention_mask,  # Multi-page attention mask.
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss, ret_loss, ret_logits = None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            # try:
            #     ret_loss, ret_logits = self.retrieval_module(document_embeddings, answer_page_idx)
            # except:
            #     ret_loss, ret_logits = self.retrieval_module(document_embeddings, answer_page_idx)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        model_output = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=page_encoder_attentions if output_attentions else None,
            # encoder_attentions=encoder_outputs.attentions,
        )

        # model_output.ret_logits = ret_logits
        # model_output.ret_loss = ret_loss

        return model_output


class Proxy_HiLT5(nn.Module):

    def __init__(self, config):
        super(Proxy_HiLT5, self).__init__()

        # self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.page_tokens = config.get('page_tokens', 20)
        self.max_doc_pages = config.get('max_pages', 20)

        config_x = CustomT5Config.from_pretrained(config['pretrained_model_name_or_path'])
        config_x.page_tokens = self.page_tokens
        config_x.max_doc_pages = self.max_doc_pages
        # config_x.page_retrieval_config = config['retrieval_module']
        self.tokenizer = T5Tokenizer.from_pretrained(config['pretrained_model_name_or_path'])
        self.tokenizer.model_max_length = config.get('max_length', 4096)
        
        self.tokenizer.add_tokens("[PAGE]")  # Single representation
        # [self.tokenizer.add_tokens("[PAGE_{:d}]".format(p)) for p in range(self.num_doc_cls_tokens)]  # Different representation

        # Whenever the number of [PAGE] tokens or Max pages per document changes, the architecture also changes and therefore, it needs to be fine-tuned.
        self.model = HiLT5.from_pretrained(config['pretrained_model_name_or_path'], config=config_x, ignore_mismatched_sizes=True)

        if config.get('freeze_encoder', False):
            for n, p in self.model.named_parameters():
                if not (n.startswith('decoder') or n.startswith('retrieval_module')):
                    p.requires_grad = False

        self.device = config['device']

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def forward(self, batch, output_attentions=False, return_pred_parse=False, return_json=False):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        words = batch['words']
        boxes = batch['boxes']
        
        # question = batch['questions']
        # context = batch['contexts']
        labels = batch['labels']
        num_pages = batch['num_pages']
        # answer_page_idx = torch.LongTensor(batch['answer_page_idx']).to(self.device)

        page_token_box = [0, 0, 1000, 1000]
        padding_box = [0, 0, 0, 0]
        eos_box = [0, 0, 0, 0]

        bs = len(input_ids)
        # if self.page_retrieval == 'oracle':
        #     raise ValueError("Oracle set-up not available for Hi-LT5. Instead, specify 'max_pages: 1' in dataset config with 'page_retrieval: custom'.")

        # elif self.page_retrieval in ['logits', 'concat']:
        #     raise ValueError("{:s} set-up not available for Hi-LT5".format(self.page_retrieval.capitalize()))

        # else:
        # input_ids, attention_mask = [], []
        # longest_sequence = 0

        """ TODO - Set the max sequence length to N(512/1024) and simplify this triplicated loop."""
        # for batch_idx in range(bs):
        #     input_text = ["{:s}: question: {:s}  context: {:s}".format("[PAGE]" * self.page_tokens, question[batch_idx], c) for c in context[batch_idx]]
        #     tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        #     input_ids.append(tokens.input_ids)
        #     attention_mask.append(tokens.attention_mask)
        #     longest_sequence = max(longest_sequence, tokens.input_ids.shape[-1])
        
        longest_sequence = max((x.shape[-1] for x in input_ids))

        all_input_ids = torch.zeros([bs, max(num_pages), longest_sequence], dtype=torch.long)
        all_attention_masks = torch.zeros([bs, max(num_pages), longest_sequence], dtype=torch.long)
        for batch_idx in range(bs):
            all_input_ids[batch_idx, :num_pages[batch_idx], :input_ids[batch_idx].shape[-1]] = input_ids[batch_idx]
            all_attention_masks[batch_idx, :num_pages[batch_idx], :attention_mask[batch_idx].shape[-1]] = attention_mask[batch_idx]

        all_boxes = torch.zeros([bs, max(num_pages), longest_sequence, 4], dtype=torch.long)
        for batch_idx in range(bs):
            for page_idx in range(num_pages[batch_idx]):
                if len(words[batch_idx][page_idx]) >= 1:
                    context_boxes = torch.tensor(np.array([box for word, word_box in zip(words[batch_idx][page_idx], boxes[batch_idx][page_idx]) for box in [word_box]*len(self.tokenizer(word).input_ids[:-1])]))
                    context_boxes = context_boxes[:self.tokenizer.model_max_length - self.page_tokens]  # Remove boxes out of model max length.
                else:
                    context_boxes = torch.tensor(padding_box)

                all_boxes[batch_idx, page_idx, :self.page_tokens] = torch.tensor(page_token_box)
                # all_boxes[batch_idx, page_idx, self.page_tokens: self.page_tokens + len(question_boxes)] = question_boxes
                all_boxes[batch_idx, page_idx, self.page_tokens: self.page_tokens + len(context_boxes)] = context_boxes
                all_boxes[batch_idx, page_idx, self.page_tokens + len(context_boxes) - 1] = torch.tensor(eos_box)

        all_input_ids = all_input_ids.to(self.device)
        all_boxes = all_boxes.to(self.device)
        all_attention_masks = all_attention_masks.to(self.device)

        labels = torch.stack(labels).to(self.device)

        # outputs = self.model(input_ids=all_input_ids, bbox=all_boxes, attention_mask=all_attention_masks, labels=labels, num_pages=num_pages, answer_page_idx=answer_page_idx)
        outputs = self.model(input_ids=all_input_ids, bbox=all_boxes, attention_mask=all_attention_masks, labels=labels, num_pages=num_pages, output_attentions=output_attentions)
        pred_parses = self.get_answer_from_model_output(all_input_ids, all_boxes, all_attention_masks, num_pages, return_json) if return_pred_parse else None
        
        # if self.page_retrieval == 'oracle':
        #     pred_answer_pages = batch['answer_page_idx']

        # else:
        #     # TODO change it from generation
        #     pred_answer_pages = outputs.ret_logits.argmax(dim=-1).tolist()

        # return outputs, pred_answers, pred_answer_pages
                
        return outputs, pred_parses

    def get_answer_from_model_output(self, input_ids, boxes, attention_mask, num_pages, return_json=False):
        output = self.model.generate(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, num_pages=num_pages, output_scores=True, return_dict_in_generate=True)
        pred_parses = []
        
        for seq in self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True):
            seq = seq.replace(self.tokenizer.eos_token, "").replace(
                self.tokenizer.pad_token, ""
            )
            seq = re.sub(
                r"<.*?>", "", seq, count=1
            ).strip()  # remove first task start token
            if return_json:
                pred_parses.append(self.token2json(seq))
            else:
                pred_parses.append(seq)

        return pred_parses

    # def inference(self, batch):
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     words = batch['words']
    #     boxes = batch['boxes']
    #     labels = batch['labels']
    #     num_pages = batch['num_pages']
        
    #     output = self.model.generate(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, num_pages=num_pages, output_scores=True, return_dict_in_generate=True)
    
    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def json2token(
        self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_special_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                        
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(
                            obj[k],
                            update_special_tokens_for_json_key,
                            sort_json_key,
                        )
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [
                    self.json2token(
                        item, update_special_tokens_for_json_key, sort_json_key
                    )
                    for item in obj
                ]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.tokenizer.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(rf"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}",
                    tokens,
                    re.IGNORECASE,
                )
                if content is not None:
                    content = content.group(1).strip()
                    if (
                        r"<s_" in content and r"</s_" in content
                    ):  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                leaf in self.decoder.tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[
                                    1:-2
                                ]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[
                    tokens.find(end_token) + len(end_token) :
                ].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(
                        tokens[6:], is_inner_value=True
                    )

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


    # def forward_record_and_retrieve_attention_dict(self, record):

    #     with torch.no_grad():
    #         batched_record = {k: [v] for k, v in record.items()}  # fake batch
    #         outputs, pred_answer, pred_answer_page = self.forward(batched_record, output_attentions=True, return_pred_answer=True)

    #     num_pages = record['num_pages']
    #     input_text = ["{:s}: question: {:s}  context: {:s}".format("[PAGE]" * self.page_tokens, record['questions'], record['contexts'][page_idx]) for page_idx in range(num_pages)]
    #     tokens = [self.tokenizer(input_text[page_idx], return_tensors='pt', padding='max_length', truncation=True) for page_idx in range(num_pages)]

    #     answers = random.choice(record['answers'])
    #     labels = self.tokenizer(answers, return_tensors='pt', padding=True)
    #     # encoder_text = model.tokenizer.convert_ids_to_tokens(tokens.input_ids[0])
    #     encoder_text = [self.tokenizer.convert_ids_to_tokens(tokens[page_idx].input_ids[0]) for page_idx in range(num_pages)]
    #     answer_text = self.tokenizer.convert_ids_to_tokens(labels.input_ids[0])
    #     decoder_input_text = ["[PAGE_{:d},{:d}]".format(page_idx, token_idx) for page_idx in range(num_pages) for token_idx in range(self.page_tokens)]

    #     # Convert tensors to CPU
    #     encoder_att = []
    #     for page_idx in range(len(outputs.encoder_attentions)):
    #         encoder_att.append([att.data.cpu() for att in outputs.encoder_attentions[page_idx]])

    #     decoder_att = [att.data.cpu() for att in outputs.decoder_attentions]
    #     cross_att = [att.data.cpu() for att in outputs.cross_attentions]

    #     att_dict = {
    #         "Page retrieval": self.page_retrieval.capitalize(),
    #         "Pred. Answer": pred_answer[0],
    #         "Pred. Answer Page": pred_answer_page[0],
    #         "question_id": record['question_id'],
    #         "doc_id": record['question_id'],
    #         "encoder_att": encoder_att,
    #         "decoder_att": decoder_att,
    #         "cross_att": cross_att,
    #         "encoder_text": encoder_text,
    #         "answer_text": answer_text,
    #         "decoder_input_text": decoder_input_text,
    #     }

    #     return outputs, pred_answer, pred_answer_page, att_dict
