import torch
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys
from data_utils.refcoco import RefCocoDataset, build_dataset
from model import ClipREGModel, ClipREGPrefix, MappingType
from os.path import join
from configuration import Config


def train(train_dataset: RefCocoDataset, val_dataset: RefCocoDataset, ciderval_dataset: RefCocoDataset, 
          model: ClipREGModel, config,
          lr: float = 2e-5, warmup_steps: int = 5000, 
          output_dir: str = ".", output_prefix: str = "", 
          device = 'cuda' if torch.cuda.is_available() else 'cpu'):

    print(f'train model on device {device}')

    batch_size = config.batch_size
    epochs = config.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    ciderval_dataloader = DataLoader(ciderval_dataset, batch_size=1, shuffle=False, drop_last=False)

    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        model.train()
        train_progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        train_losses = []
        for idx, (ann_id, *encoder_input, tokens, mask) in enumerate(train_dataloader):
            model.zero_grad()
            target, context, loc = encoder_input
            tokens, mask, target, context, loc = tokens.to(device), mask.to(device), target.to(device, dtype=torch.float32), context.to(device, dtype=torch.float32), loc.to(device, dtype=torch.float32)
            outputs = model(tokens, target=target, context=context, loc=loc, mask=mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            train_progress.set_postfix({"loss": loss.item()})
            train_progress.update()
        train_progress.close()
        train_loss = torch.tensor(train_losses).mean().item()
        print(f'Train loss: {train_loss}')

        model.eval()
        print(f">>> Evaluation epoch {epoch}")
        val_progress = tqdm(total=len(val_dataloader), desc=output_prefix)
        val_losses = []
        for idx, (ann_id, *encoder_input, tokens, mask) in enumerate(val_dataloader):
            target, context, loc = encoder_input
            tokens, mask, target, context, loc = tokens.to(device), mask.to(device), target.to(device, dtype=torch.float32), context.to(device, dtype=torch.float32), loc.to(device, dtype=torch.float32)
            outputs = model(tokens, target=target, context=context, loc=loc, mask=mask)
            logits = outputs.logits[:, val_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            val_losses.append(loss.item())
            val_progress.set_postfix({"loss": loss.item()})
            val_progress.update()
        val_progress.close()
        val_loss = torch.tensor(val_losses).mean().item()
        print(f'Val loss: {val_loss}')

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
        model = ClipREGPrefix(prefix_length, clip_length=config.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=config.num_layers, mapping_type=config.mapping_type)
        print(f'Built {model.__class__.__name__} model')
        print("Train only prefix")
    else:
        model = ClipREGModel(prefix_length, clip_length=config.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=config.num_layers, mapping_type=config.mapping_type)
        print(f'Built {model.__class__.__name__} model')
        print("Train both prefix and GPT")
        sys.stdout.flush()

    train_dataset = build_dataset(
        transform=model.backbone.preprocess,
        tokenizer=model.tokenizer,
        ref_dir= join(config.ref_base, config.dataset),
        coco_dir=config.coco_dir,
        verbose=config.verbose,
        prefix_length=model.prefix_length,
        mode='training'
    )

    val_dataset = build_dataset(
        transform=model.backbone.preprocess,
        tokenizer=model.tokenizer,
        ref_dir= join(config.ref_base, config.dataset),
        coco_dir=config.coco_dir,
        verbose=config.verbose,
        prefix_length=model.prefix_length,
        mode='val'
    )

    ciderval_dataset = build_dataset(
        transform=model.backbone.preprocess,
        tokenizer=model.tokenizer,
        ref_dir= join(config.ref_base, config.dataset),
        coco_dir=config.coco_dir,
        verbose=config.verbose,
        prefix_length=model.prefix_length,
        mode='val',
        return_unique=True
    )    

    train(train_dataset, val_dataset, ciderval_dataset, model, config, output_dir=config.checkpoint_dir, output_prefix=config.output_prefix)


if __name__ == '__main__':
    main()
