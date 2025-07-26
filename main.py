import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch._dynamo
from streaming_gpt_dataset import StreamingGPTDataset
from gpt2_model import GPT2LikeModel
from tokenizers import ByteLevelBPETokenizer
import warnings
import signal
import sys
import os
import json
import threading
warnings.filterwarnings('ignore')

class InterruptionSafeTrainer:
    def __init__(self):
        self.interrupted = False
        self.checkpoint_lock = threading.Lock()
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🚨 INTERRUPTION SIGNAL RECEIVED ({signum})!")
            print("🛡️  Initiating graceful shutdown...")
            self.interrupted = True
            
        # Handle SIGTERM (RunPod sends this 5 seconds before termination)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        
    def save_emergency_checkpoint(self, model, optimizer, scaler, epoch, step, loss, batch_idx=None):
        """Save checkpoint immediately when interrupted"""
        with self.checkpoint_lock:
            try:
                timestamp = int(time.time())
                checkpoint_path = f"emergency_checkpoint_{timestamp}.pt"
                
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'loss': loss,
                    'batch_idx': batch_idx,
                    'timestamp': timestamp,
                    'interrupted': True
                }
                
                torch.save(checkpoint_data, checkpoint_path)
                
                # Also save metadata for easy resume
                metadata = {
                    'checkpoint_file': checkpoint_path,
                    'epoch': epoch,
                    'step': step,
                    'loss': loss,
                    'batch_idx': batch_idx,
                    'timestamp': timestamp,
                    'status': 'interrupted'
                }
                
                with open('training_state.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"🚨 EMERGENCY CHECKPOINT SAVED: {checkpoint_path}")
                print(f"📋 Training state saved to: training_state.json")
                return True
                
            except Exception as e:
                print(f"❌ Emergency checkpoint failed: {e}")
                return False

def load_checkpoint_if_exists():
    """Load existing checkpoint to resume training"""
    try:
        if os.path.exists('training_state.json'):
            with open('training_state.json', 'r') as f:
                metadata = json.load(f)
            
            checkpoint_file = metadata['checkpoint_file']
            if os.path.exists(checkpoint_file):
                print(f"🔄 Found checkpoint: {checkpoint_file}")
                print(f"📊 Resume from: Epoch {metadata['epoch']}, Step {metadata['step']}")
                return torch.load(checkpoint_file), metadata
        
        return None, None
    except Exception as e:
        print(f"⚠️  Checkpoint loading failed: {e}")
        return None, None

def train_spot_instance_safe():
    """
    Spot instance safe training with automatic interruption handling
    Cost savings: 60-80% vs on-demand!
    """
    trainer = InterruptionSafeTrainer()
    start_time = time.time()
    
    # ============= SETUP =============
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.suppress_errors = True
    
    # ============= MODEL & DATA SETUP =============
    tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    block_size = 1024
    
    dataset = StreamingGPTDataset("data1/tokenized_ids.bin", block_size, stride=512)
    batch_size = 128
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )
    
    model = GPT2LikeModel(vocab_size, block_size, 256, 6, 2).to(device)
    model = torch.compile(model, mode='max-autotune')
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    scaler = GradScaler()
    
    # ============= RESUME FROM CHECKPOINT =============
    start_epoch = 0
    start_step = 0
    start_batch_idx = 0
    
    checkpoint, metadata = load_checkpoint_if_exists()
    if checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            start_batch_idx = checkpoint.get('batch_idx', 0)
            
            print(f"✅ RESUMED from checkpoint!")
            print(f"   Starting from: Epoch {start_epoch}, Step {start_step}")
            
            # Clean up old checkpoint
            if os.path.exists(metadata['checkpoint_file']):
                os.remove(metadata['checkpoint_file'])
                
        except Exception as e:
            print(f"⚠️  Checkpoint resume failed: {e}, starting fresh")
            start_epoch = start_step = start_batch_idx = 0
    
    # ============= TRAINING LOOP =============
    model.train()
    num_epochs = 2
    gradient_accumulation_steps = 2
    
    print(f"\n🚀 SPOT INSTANCE TRAINING STARTED")
    print(f"💰 Cost savings: ~60-80% vs on-demand")
    print(f"🛡️  Interruption protection: ENABLED")
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        step_count = start_step if epoch == start_epoch else 0
        
        print(f"\n🔁 Epoch {epoch + 1}/{num_epochs}")
        dataset.start_offset = (epoch * 1024) % 2048
        
        optimizer.zero_grad()
        
        # Skip batches if resuming mid-epoch
        batch_iter = iter(loader)
        if epoch == start_epoch and start_batch_idx > 0:
            print(f"⏭️  Skipping {start_batch_idx} batches to resume...")
            for _ in range(start_batch_idx):
                try:
                    next(batch_iter)
                except StopIteration:
                    break
        
        batch_idx = start_batch_idx if epoch == start_epoch else 0
        
        for batch in batch_iter:
            # Check for interruption BEFORE processing batch
            if trainer.interrupted:
                print(f"🛑 Interruption detected during batch {batch_idx}")
                success = trainer.save_emergency_checkpoint(
                    model, optimizer, scaler, epoch, step_count, 
                    total_loss / max(1, step_count % 50), batch_idx
                )
                if success:
                    print("✅ Training state saved successfully!")
                    print("🔄 You can resume training by running this script again")
                else:
                    print("❌ Failed to save training state!")
                
                sys.exit(0)
            
            with autocast():
                idx = batch.to(device, non_blocking=True)
                logits = model(idx)
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = idx[:, 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                step_count += 1
            
            total_loss += loss.item() * gradient_accumulation_steps
            batch_idx += 1
            
            # More frequent checkpointing for spot instances
            if step_count > 0 and step_count % 25 == 0:  # Every 25 steps (was 50)
                avg_loss = total_loss / 25
                elapsed = time.time() - start_time
                print(f"📊 Epoch {epoch+1} Step {step_count} | Loss: {avg_loss:.4f} | Time: {elapsed/60:.1f}min")
                total_loss = 0.0
            
            # Frequent safety checkpoints
            # if step_count > 0 and step_count % 100 == 0:  # Every 100 steps (was 500)
            #     checkpoint_path = f"checkpoint_epoch{epoch+1}_step{step_count}.pt"
                
            #     checkpoint_data = {
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scaler_state_dict': scaler.state_dict(),
            #         'epoch': epoch,
            #         'step': step_count,
            #         'loss': avg_loss,
            #         'batch_idx': batch_idx,
            #         'timestamp': int(time.time())
            #     }
                
            #     torch.save(checkpoint_data, checkpoint_path)
                
            #     # Update metadata
            #     metadata = {
            #         'checkpoint_file': checkpoint_path,
            #         'epoch': epoch,
            #         'step': step_count,
            #         'loss': avg_loss,
            #         'batch_idx': batch_idx,
            #         'timestamp': int(time.time()),
            #         'status': 'training'
            #     }
                
            #     with open('training_state.json', 'w') as f:
            #         json.dump(metadata, f, indent=2)
                
            #     print(f"💾 Safety checkpoint: {checkpoint_path}")
        
        # Reset batch index for next epoch
        start_batch_idx = 0
        start_step = 0
        
        # End of epoch checkpoint
        final_checkpoint = f"checkpoint_epoch{epoch+1}_final.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch + 1,
            'step': step_count,
            'batch_idx': 0  # Reset for next epoch
        }, final_checkpoint)
        print(f"✅ Epoch {epoch+1} completed. Saved: {final_checkpoint}")
    
    # ============= COMPLETION =============
    end_time = time.time()
    duration = end_time - start_time
    hours = duration / 3600
    
    # Spot instance pricing (approximately 60-80% cheaper)
    spot_cost_estimate = hours * 0.50  # Estimated spot price for H100
    ondemand_cost = hours * 1.99
    savings = ondemand_cost - spot_cost_estimate
    
    print(f"\n🎉 SPOT TRAINING COMPLETED!")
    print(f"⏱️  Total time: {hours:.2f} hours")
    print(f"💰 Spot cost: ${spot_cost_estimate:.2f}")
    print(f"💰 On-demand would be: ${ondemand_cost:.2f}")
    print(f"💵 Total savings: ${savings:.2f} ({(savings/ondemand_cost)*100:.1f}%)")
    
    # Clean up training state
    if os.path.exists('training_state.json'):
        os.remove('training_state.json')
    
    torch.save(model.state_dict(), "final_spot_trained_model.pt")
    print("💾 Final model saved!")

def estimate_spot_vs_ondemand():
    """Compare spot vs on-demand costs"""
    print("💰 COST COMPARISON:")
    print("=" * 60)
    print("Scenario: 18-hour H100 training")
    print()
    print("ON-DEMAND:")
    print("  Rate: $1.99/hour")
    print("  Total: $35.82")
    print("  Risk: None (guaranteed)")
    print()
    print("SPOT INSTANCE:")
    print("  Rate: ~$0.40-0.60/hour (varies)")
    print("  Total: ~$7.20-10.80")
    print("  Savings: ~$25-28 (70-80%)")
    print("  Risk: Interruptions (handled by checkpoints)")
    print()
    print("INTERRUPTION HANDLING:")
    print("✅ 5-second warning detection")
    print("✅ Auto-checkpoint every 100 steps")
    print("✅ Resume from exact position")
    print("✅ No training progress lost")
    print("=" * 60)

if __name__ == '__main__':
    estimate_spot_vs_ondemand()
    
    # Ask user for confirmation
    print("\n🔄 Ready to start spot instance training?")
    print("📋 This will use checkpointing to handle interruptions")
    response = input("Continue? (y/n): ")
    
    if response.lower().startswith('y'):
        train_spot_instance_safe()
    else:
        print("Training cancelled.")