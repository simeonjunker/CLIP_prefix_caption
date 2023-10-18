import torch
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys
from data_utils.refcoco import RefCocoDataset, build_dataset
from model import ClipCaptionModel, ClipCaptionPrefix, MappingType
from os.path import join
from configuration import Config


def train(dataset: RefCocoDataset, model: ClipCaptionModel, config,
          lr: float = 2e-5, warmup_steps: int = 5000, 
          output_dir: str = ".", output_prefix: str = "", 
          device = 'cuda' if torch.cuda.is_available() else 'cpu'):

    print(f'train model on device {device}')

    batch_size = config.batch_size
    epochs = config.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)

        for idx, (ann_id, *encoder_input, tokens, mask) in enumerate(train_dataloader):
            model.zero_grad()
            target, context, loc = encoder_input
            tokens, mask, target = tokens.to(device), mask.to(device), target.to(device, dtype=torch.float32)
            outputs = model(tokens, target, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % config.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model


def main():

    config = Config()

    prefix_length = config.prefix_length
    prefix_dim = 640 if config.is_rn else 512
    config.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[config.mapping_type]
    if config.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=config.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=config.num_layers, mapping_type=config.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=config.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=config.num_layers, mapping_type=config.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()

    dataset = build_dataset(
        transform=model.backbone.preprocess,
        tokenizer=model.tokenizer,
        ref_dir= join(config.ref_base, config.dataset),
        coco_dir=config.coco_dir,
        verbose=config.verbose,
        mode='training'
    )

    train(dataset, model, config, output_dir=config.checkpoint_dir, output_prefix=config.output_prefix)


if __name__ == '__main__':
    main()
