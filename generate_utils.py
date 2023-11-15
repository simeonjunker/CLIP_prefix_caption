import numpy as np
import torch
import torch.nn.functional as nnf

def generate_beam(model, 
                  tokenizer, 
                  beam_size: int = 5, 
                  embed=None,
                  entry_length=67, 
                  temperature=1., 
                  stop_token=None,
                  set_to_eval=True
    ):

    if set_to_eval:
        model.eval()
    stop_token = tokenizer.eos_token if stop_token is None else stop_token
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    grad_context = torch.no_grad if set_to_eval else torch.enable_grad
    with grad_context():
        generated = embed
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)], skip_special_tokens=True) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts, output_list


def batch_greedy(model, 
                 tokenizer, 
                 embed, 
                 entry_length=67, 
                 stop_token=None,
                 set_to_eval=True):
    """greedy decoding for a batch of samples"""
        
    if set_to_eval:
        model.eval()

    device = next(model.parameters()).device
    
    stop_token = tokenizer.eos_token if stop_token is None else stop_token
    stop_token_index = tokenizer.encode(stop_token)[0]
    finished = torch.zeros((embed.shape[0], 1), dtype=bool, device=device) # type: ignore
    all_logits = []
    caption = mask = torch.zeros((embed.shape[0], 0), dtype=int) # type: ignore


    grad_context = torch.no_grad if set_to_eval else torch.enable_grad
    with grad_context():
        generated = embed

        for i in range(entry_length):
            
            mask = torch.cat([mask, finished], 1)
            
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :]
            all_logits.append(logits.unsqueeze(0))
            
            next_tokens = torch.argmax(logits, -1).unsqueeze(1)
            next_tokens_embed = model.gpt.transformer.wte(next_tokens)
            generated = torch.cat((generated, next_tokens_embed), dim=1)
            
            caption = torch.cat([caption, next_tokens], 1)
                    
            is_eos = next_tokens == stop_token_index
            finished = torch.logical_or(is_eos, finished)
            caption[:, -1:][finished] = stop_token_index
            if all(finished):
                break
        
    output_text = tokenizer.batch_decode(caption, skip_special_tokens=True)
    all_logits = torch.cat(all_logits)
        
    return output_text, caption, all_logits, mask.bool()


def generate_greedy(
        model,
        tokenizer,
        tokens=None,
        embed=None,
        entry_length=67,  # maximum number of words
        stop_token=None,
        set_to_eval=True
        ):
    
    if set_to_eval:
        model.eval()
    stop_token = tokenizer.eos_token if stop_token is None else stop_token
    stop_token_index = tokenizer.encode(stop_token)[0]
    device = next(model.parameters()).device
    all_logits = []

    grad_context = torch.no_grad if set_to_eval else torch.enable_grad
    with grad_context():
        generated = embed

        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :]
            all_logits.append(logits)
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break

        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list, skip_special_tokens=True)
        all_logits = torch.cat(all_logits).unsqueeze(0)

    return output_text, output_list, all_logits


def generate_topp(
        model,
        tokenizer,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token=None,
        set_to_eval=True,
):
    if set_to_eval:
        model.eval()
    generated_tokens = []
    generated_text = []
    stop_token = tokenizer.eos_token if stop_token is None else stop_token
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    all_logits = []

    grad_context = torch.no_grad if set_to_eval else torch.enable_grad
    with grad_context():

        for entry_idx in range(entry_count):
            generated = embed
            tokens = None

            entry_logits = []

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :]
                entry_logits.append(logits)
                logits = logits / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                
                
                probs = logits.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            all_logits.append(torch.cat(entry_logits).unsqueeze(0))
            
            output_list = list(tokens.squeeze().cpu().numpy())
            generated_tokens.append(output_list)
            output_text = tokenizer.decode(output_list, skip_special_tokens=True)
            generated_text.append(output_text)

    return generated_text, generated_tokens, all_logits


def batch_topp(
    model, 
    tokenizer, 
    embed, 
    entry_count=2, 
    entry_length=67,
    top_p=0.8,
    temperature=1., 
    device="auto", 
    stop_token=None,
    set_to_eval=True
    ):
    """greedy decoding for a batch of samples"""
        
    if set_to_eval:
        model.eval()

        
    device = next(model.parameters()).device
    
    stop_token = tokenizer.eos_token if stop_token is None else stop_token
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    all_logits = []
    all_captions = []
    all_masks = []
    
    grad_context = torch.no_grad if set_to_eval else torch.enable_grad
    with grad_context():
    
        for entry_idx in range(entry_count):
            generated = embed
            finished = torch.zeros((embed.shape[0], 1), dtype=bool, device=device) # type: ignore
            caption = mask = torch.zeros((embed.shape[0], 0), dtype=int) # type: ignore
            entry_logits = []

            for i in range(entry_length):
                
                mask = torch.cat([mask, finished], 1)
                
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :]
                entry_logits.append(logits.unsqueeze(0))
                logits = logits / (temperature if temperature > 0 else 1.0)
                            
                            
                sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for sample_idx, (si, si_r) in enumerate(zip(sorted_indices, sorted_indices_to_remove)):
                    indices_to_remove = si[si_r]
                    logits[sample_idx, indices_to_remove] = filter_value
                                        
                probs = nnf.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                next_tokens_embed = model.gpt.transformer.wte(next_tokens)
                generated = torch.cat((generated, next_tokens_embed), dim=1)

                caption = torch.cat([caption, next_tokens], 1)
                        
                is_eos = next_tokens == stop_token_index
                finished = torch.logical_or(is_eos, finished)
                caption[:, -1:][finished] = stop_token_index
                if all(finished):
                    break
                        
            all_logits.append(entry_logits)
            all_captions.append(caption)
            all_masks.append(mask.bool())
        
    all_texts = [
        tokenizer.batch_decode(caption, skip_special_tokens=True)
        for caption in all_captions
    ]

    all_logits_concat = [
        torch.cat(al) for al in all_logits
    ]
        
    return all_texts, all_captions, all_logits_concat, all_masks