import argparse
import os
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from evaluate import load
import torch

from dataset import TrOCRDataset


def compute_metrics(pred, processor):
    cer_metric = load("cer")
    
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}


def main(args):
    print("="*60)
    print("TrOCR Fine-tuning - Hungarian Documents")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nLoading model: {args.model_name}")
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    print("\nLoading datasets...")
    train_dataset = TrOCRDataset(
        csv_path=args.train_csv,
        images_dir=args.images_dir,
        processor=processor,
        max_length=args.max_length
    )
    
    val_dataset = TrOCRDataset(
        csv_path=args.val_csv,
        images_dir=args.images_dir,
        processor=processor,
        max_length=args.max_length
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="tensorboard",
        predict_with_generate=True,
        generation_max_length=args.max_length,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=default_data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\n" + "="*60)
    print("Training complete! Saving model...")
    print("="*60)
    
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print(f"\n✓ Model saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--train_csv", type=str, default="data/processed/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/processed/val.csv")
    parser.add_argument("--images_dir", type=str, default="data/processed/images")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="models/base-hungarian")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)