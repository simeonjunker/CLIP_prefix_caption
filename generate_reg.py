from pathlib import Path
import argparse
import torch
from tqdm import tqdm
import os.path as osp
from configuration import Config
from model import ClipREGModel, ClipREGPrefix, MappingType
from data_utils.refcoco import build_dataset
from generate_utils import generate_beam, generate2
import json


def main(args, config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # make model
    prefix_length = config.prefix_length
    prefix_dim = 640 if config.is_rn else 512
    config.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[config.mapping_type]
    if config.only_prefix:
        model = ClipREGPrefix(prefix_length, clip_length=config.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=config.num_layers, mapping_type=config.mapping_type)
        print(f'Built {model.__class__.__name__} model')
    else:
        model = ClipREGModel(prefix_length, clip_length=config.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=config.num_layers, mapping_type=config.mapping_type)
        print(f'Built {model.__class__.__name__} model')

    model.load_state_dict(torch.load(args.model_checkpoint, map_location="cpu")) 
    model.to(device)
    model.eval()

    # make dataset
    dataset = build_dataset(
        transform=model.backbone.preprocess,
        tokenizer=model.tokenizer,
        ref_dir= str(Path(config.ref_base, config.dataset).resolve()),
        coco_dir=config.coco_dir,
        verbose=config.verbose,
        prefix_length=model.prefix_length,
        return_unique=True,
        mode=args.split
    )

    if args.decoding_method == 'beam':
        print('using beam search')
        def generate(model, tokenizer, embed): 
            return generate_beam(model, tokenizer, embed=embed)[0][0]
    else:
        print('using greedy search')
        def generate(model, tokenizer, embed):
            return generate2(model, tokenizer, embed=embed)[0]

    results = []

    for i, (ann_id, *encoder_input, tokens, mask) in tqdm(enumerate(dataset)):

        if i > 10:
            break

        target, context, loc = encoder_input
        target, context, loc = (
            target.to(device, dtype=torch.float32).unsqueeze(0), 
            context.to(device, dtype=torch.float32).unsqueeze(0), 
            loc.to(device, dtype=torch.float32).unsqueeze(0)
        )

        prefix_embed = model.make_visual_prefix(
            target, context, loc).reshape(1, prefix_length, -1)
        
        generated = generate(model, model.tokenizer, prefix_embed)

        results.append({
            'ann_id': ann_id,
            'generated': generated
        })

    model_name = osp.split(args.model_checkpoint)[-1].replace('.pt', '')
    out_file = str(Path(args.out_dir, f'{model_name}_{args.split}_{args.decoding_method}.json').resolve())

    with open(out_file, 'w') as f:
        print(f'write results to {out_file}')
        json.dump(results, f)
        

if __name__ == '__main__':
    config = Config()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_checkpoint', required=True)
    parser.add_argument('--decoding_method', default='greedy', choices=['greedy', 'beam'], type=str.lower)
    parser.add_argument('--out_dir', default='./generated')
    parser.add_argument('--split', default='val', choices=['val', 'testa', 'testa'], type=str.lower)

    args = parser.parse_args()


    # make sure the output dir exists
    p = Path(args.out_dir).expanduser().resolve()
    if not p.exists():
        print(f'Create output directory at {str(p)}')
        p.mkdir()
    args.out_dir = str(p)

    # make sure the checkpoint exists
    assert Path(args.model_checkpoint).expanduser().exists(), f'checkpoint {args.model_checkpoint} does not exist'

    main(args, config)