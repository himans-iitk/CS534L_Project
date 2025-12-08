from .dataset import DefaultDataset
from .utils import load_model_and_tokenizer

import transformers


class EpochProgressCallback(transformers.TrainerCallback):
    """Callback to print epoch progress and loss."""
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.epoch_losses = []  # Store losses for each epoch
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch) + 1
        print(f"\n{'='*60}")
        print(f"Starting Epoch {self.current_epoch}/{self.total_epochs}")
        print(f"{'='*60}")
    
    def on_log(self, args, state, control, **kwargs):
        """Capture loss during training."""
        if hasattr(state, 'log_history') and state.log_history:
            # Get the latest log entry
            latest_log = state.log_history[-1]
            if 'loss' in latest_log and 'epoch' in latest_log:
                epoch_num = int(latest_log['epoch'])
                if epoch_num == self.current_epoch - 1:  # Current epoch
                    # Store the loss for this epoch
                    if len(self.epoch_losses) < self.current_epoch:
                        self.epoch_losses.append([])
                    if len(self.epoch_losses) == self.current_epoch - 1:
                        self.epoch_losses.append([])
                    self.epoch_losses[self.current_epoch - 1].append(latest_log['loss'])
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Print epoch completion with loss information."""
        print(f"\n✓ Completed Epoch {self.current_epoch}/{self.total_epochs}")
        
        # Try multiple ways to get the loss
        epoch_loss = None
        
        # Method 1: From log_history
        if hasattr(state, 'log_history') and state.log_history:
            # Look for the most recent log entry for this epoch
            for log_entry in reversed(state.log_history):
                if 'loss' in log_entry and 'epoch' in log_entry:
                    log_epoch = log_entry.get('epoch', 0)
                    if abs(log_epoch - (self.current_epoch - 1)) < 0.1:  # Current epoch
                        epoch_loss = log_entry.get('loss')
                        break
        
        # Method 2: From stored losses
        if epoch_loss is None and len(self.epoch_losses) >= self.current_epoch:
            epoch_loss_list = self.epoch_losses[self.current_epoch - 1]
            if epoch_loss_list:
                epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)  # Average loss
        
        # Method 3: From training metrics
        if epoch_loss is None and hasattr(state, 'train_metrics'):
            epoch_loss = state.train_metrics.get('train_loss')
        
        if epoch_loss is not None:
            print(f"  📉 Epoch Loss: {epoch_loss:.6f}")
        else:
            print(f"  ⚠ Loss information not available")
        
        print(f"{'='*60}\n")


def finetune(
    model_dir: str,
    data_file: str,
    out_dir: str,
    epochs: int = 5,
    per_device_batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    gradient_accumulation_steps: int = 1
):
    import torch
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared GPU cache")
    
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled (saves memory)")
    model.config.use_cache = False  # Required when using gradient checkpointing

    dataset = DefaultDataset(
        data_file,
        tokenizer=tokenizer,
        max_len=max_len
    )

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        bf16=True,
        report_to='none',        # Disable wandb
        save_strategy='no',      # Disable checkpoint saving to reduce memory
        save_steps=0,            # Disable checkpoint saving
        save_total_limit=0,      # Don't keep any checkpoints
        # Memory optimization settings
        dataloader_pin_memory=False,  # Disable pin_memory to save memory
        dataloader_num_workers=0,     # Use single process to save memory
        remove_unused_columns=False,  # Keep columns as-is
        logging_steps=10,             # Log more frequently to capture loss
        logging_first_step=True,      # Log the first step
    )

    # Create callback for epoch progress
    epoch_callback = EpochProgressCallback(total_epochs=epochs)

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        callbacks=[epoch_callback]  # Add epoch progress callback
    )
    
    # Check GPU memory before training
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"\n📊 GPU Memory: {allocated:.2f} GB / {gpu_memory:.2f} GB allocated")
    
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    print(f"\n🚀 Starting finetuning for {epochs} epochs...")
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Batch size per device: {per_device_batch_size}")
    print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Model will be saved only at the end (no checkpoints)")
    print(f"\n💡 If you get CUDA OOM errors, try:")
    print(f"   - Reducing batch size (currently: {per_device_batch_size})")
    print(f"   - Increasing gradient_accumulation_steps (currently: {gradient_accumulation_steps})")
    print(f"   - Reducing max_len (currently: {max_len})\n")
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"\n❌ CUDA Out of Memory Error!")
            print(f"\n💡 Solutions to try:")
            print(f"   1. Reduce batch size: per_device_batch_size={max(1, per_device_batch_size // 2)}")
            print(f"   2. Increase gradient accumulation: gradient_accumulation_steps={gradient_accumulation_steps * 2}")
            print(f"   3. Reduce max_len: max_len={max(512, max_len // 2)}")
            print(f"   4. Clear GPU cache and try again")
            raise
        else:
            raise
    
    # Clear GPU cache before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n💾 Saving final model to {out_dir}...")
    trainer.save_model(out_dir)
    
    # Final loss summary
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        final_losses = []
        for log_entry in trainer.state.log_history:
            if 'loss' in log_entry and 'epoch' in log_entry:
                final_losses.append((log_entry.get('epoch', 0), log_entry.get('loss')))
        
        if final_losses:
            print(f"\n📊 Training Loss Summary:")
            for epoch_num, loss_val in final_losses:
                if loss_val is not None:
                    print(f"   Epoch {int(epoch_num) + 1}: Loss = {loss_val:.6f}")
    
    print(f"\n✓ Finetuning completed! Final model saved to: {out_dir}")
    
    # Final memory check
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"📊 Final GPU Memory: {allocated:.2f} GB allocated")
